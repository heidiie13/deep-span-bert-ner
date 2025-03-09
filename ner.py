import argparse
import logging
import sys
import os
import datetime
import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from dsbert.models.deep_span_extractor import DeepSpanExtractorConfig
from dsbert.models.deep_span_classification import DeepSpanClsDecoderConfig
from dsbert.models.encoders.bert_like import BertLikeConfig
from dsbert.models.encoders.span_bert_like import SpanBertLikeConfig
from dsbert.training.trainer import Trainer
from dsbert.training.evaluation import evaluate_entity_recognition
from dsbert.dataset import Dataset
from dsbert.utils import load_data, load_pretrained, count_parameters

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="phoner_covid19")
    
    parser.add_argument("--bert_arch", type=str, default="vinai/phobert-base")
    parser.add_argument("--bert_drop_rate", type=float, default=0.2)
    parser.add_argument("--bert_freeze", action="store_true")
    
    parser.add_argument("--max_span_size", type=int, default=None)
    parser.add_argument("--init_drop_rate", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=None)
    
    parser.add_argument("--no_share_weights_ext", dest='share_weights_ext', action='store_false', default=True)
    parser.add_argument("--no_share_weights_int", dest='share_weights_int', action='store_false', default=True)
    parser.add_argument("--no_share_interm2", dest='share_interm2', action='store_false', default=True)
    
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_by_loss", action="store_true", default=False)
    return parser.parse_args()

def setup_logger(save_path):
    os.makedirs(save_path, exist_ok=True)
    
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{save_path}/training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def initialize_datasets_and_config(args: argparse.Namespace):
    train_data, dev_data, test_data = load_data(args.dataset)
    bert_model, tokenizer = load_pretrained(args.bert_arch, args.bert_drop_rate)
    
    bert_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert_model, freeze=args.bert_freeze, arch=args.bert_arch)
    span_config = SpanBertLikeConfig(bert_like=bert_model,
                                     freeze=args.bert_freeze, 
                                     init_drop_rate=args.init_drop_rate,
                                     share_weights_ext=args.share_weights_ext, 
                                     share_weights_int=args.share_weights_int, 
                                     num_layers=args.num_layers)
    decoder_config = DeepSpanClsDecoderConfig()
    extractor_config = DeepSpanExtractorConfig(
        decoder=decoder_config,
        bert_like=bert_config,
        span_bert_like=span_config,
        share_interm2=args.share_interm2,
        max_span_size=args.max_span_size
    )
    
    train_dataset = Dataset(train_data, extractor_config)
    dev_dataset = Dataset(dev_data, extractor_config)
    test_dataset = Dataset(test_data, extractor_config)
    
    train_dataset.build_vocabs(dev_data, test_data)

    return train_dataset, dev_dataset, test_dataset, extractor_config

def build_model(extractor_config):
    model = extractor_config.instantiate()
    num_params, trainable_params = count_parameters(model)
    return model, num_params, trainable_params

def train_model(args, model, train_dataset, dev_dataset, save_path, device):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dev_dataset.collate)

    remaining_params = [p for p in model.parameters() if p not in set(model.pretrained_parameters())]
    
    optimizer = optim.AdamW([
        {'params': remaining_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': model.pretrained_parameters(), 'lr': args.finetune_lr, 'weight_decay': 0.0}
    ])    

    steps_per_epoch = (len(train_dataloader) + args.accumulation_steps - 1) // args.accumulation_steps
    num_warmup_steps = max(2, args.num_epochs // 5) * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, steps_per_epoch * args.num_epochs)
    
    trainer = Trainer(model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip, use_amp=args.use_amp, device=device)

    torch.save(extractor_config, f"{save_path}/dsbert_config.pth")
    trainer.train(
        train_dataloader, dev_dataloader,
        num_epochs=args.num_epochs, accumulation_steps=args.accumulation_steps,
        early_stop_patience=args.early_stop_patience, checkpoint_path=f"{save_path}/dsbert_best_model.pth",
        save_by_loss=args.save_by_loss
    )
    return trainer

def load_model_and_config_for_evaluation(config_path, model_path, device):
    model_config = torch.load(config_path, weights_only=False)
    bert_model, _ = load_pretrained(model_config.bert_like.arch, args.bert_drop_rate)
    
    model_config.bert_like.bert_like = bert_model
    model_config.span_bert_like.bert_like = bert_model
    model = model_config.instantiate()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    return model, model_config

def evaluate_model(trainer, test_dataset):
    evaluate_entity_recognition(trainer, test_dataset)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"cache/{args.dataset}/{timestamp}"
    logger = setup_logger(save_path)

    logger.info("============= Starting =============")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(args.__dict__)

    train_dataset, dev_dataset, test_dataset, extractor_config = initialize_datasets_and_config(args)
    logger.info(f"Summary train dataset: \n{train_dataset.summary}")
    model, num_params, trainable_params = build_model(extractor_config)
    
    logger.info(f"Number of parameters: {num_params}, trainable parameters: {trainable_params}")

    trainer = train_model(args, model, train_dataset, dev_dataset, save_path, device)

    logger.info("============= Evaluation =============")
    model,_ = load_model_and_config_for_evaluation(f"{save_path}/dsbert_config.pth", f"{save_path}/dsbert_best_model.pth", device)
    trainer = Trainer(model, optimizer=None, device=device)
    evaluate_model(trainer, test_dataset)
    logger.info("============= Ending =============")
