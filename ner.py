import argparse
import logging
import sys
import os
import datetime
import numpy, random
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    numpy.random.seed(seed)
    random.seed(seed)
    
def parse_to_args(parser: argparse.ArgumentParser):
    parser.add_argument("--bert_arch", type=str, default="roberta-base", help="bert-like architecture")
    parser.add_argument("--dataset", type=str, default="phoner_covid19", help="dataset name")
    parser.add_argument("--bert_freeze", default=False, action="store_true", help="bert-like freeze")
    parser.add_argument("--max_span_size", type=int, default=None, help="max_span_size")
    parser.add_argument("--num_layers", type=int, default=None, help="num_layers of span_bert_like")
    parser.add_argument('--no_share_weights_ext', dest='share_weights_ext', default=True, action='store_false', 
                               help="whether to share weights between span-bert and bert encoders")
    parser.add_argument('--no_share_weights_int', dest='share_weights_int', default=True, action='store_false', 
                               help="whether to share weights across span-bert encoders")
    parser.add_argument("--num_epochs", type=int, default=10, help="num_epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="grad_clip")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--finetune_lr", type=float, default=2e-5, help="finetune learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay")
    parser.add_argument("--use_amp", default=False, action="store_true", help="use_amp")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="early_stop_patience")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="accumulation_steps")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    
    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_to_args(parser)
    
    set_seed(args.seed)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path = f"cache/{args.dataset}/{timestamp}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
            
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        
        handlers=[
            logging.FileHandler(f"{save_path}/training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("=============Starting=============")
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    logger.info(args.__dict__)

    train_data, dev_data, test_data = load_data(args.dataset)
    bert_model, tokenizer = load_pretrained(args.bert_arch)

    bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert_model, freeze=args.bert_freeze)
    span_bert_like_config = SpanBertLikeConfig(bert_like=bert_model, freeze=args.bert_freeze, share_weights_ext=args.share_weights_ext, share_weights_int=args.share_weights_int, num_layers=args.num_layers)
    decoder_config = DeepSpanClsDecoderConfig()
    extractor_config = DeepSpanExtractorConfig(
        decoder=decoder_config,
        bert_like=bert_like_config,
        span_bert_like=span_bert_like_config,
        max_span_size=args.max_span_size
    )

    train_dataset = Dataset(train_data, extractor_config)
    dev_dataset = Dataset(dev_data, extractor_config)
    test_dataset = Dataset(test_data, extractor_config)

    train_dataset.build_vocabs(dev_data, test_data)

    logger.info(f"\nSummary train dataset: \n{train_dataset.summary}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle= True, collate_fn=train_dataset.collate)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=dev_dataset.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= False, collate_fn=dev_dataset.collate)

    model = extractor_config.instantiate()
    num_params, trainable_params = count_parameters(model)
    logger.info(f"Number of parameters: {num_params}, trainable parameters: {trainable_params}")

    remaining_params = [p for p in model.parameters() if p not in set(model.pretrained_parameters())]

    optimizer = optim.Adam([
    {'params': remaining_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
    {'params': model.pretrained_parameters(), 'lr': args.finetune_lr, 'weight_decay': args.weight_decay}
    ])    
    
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    num_training_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    trainer = Trainer(model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip, use_amp=True, device=device)

    torch.save(extractor_config, f"{save_path}/dsbert_config.pth")
    trainer.train(train_dataloader, dev_dataloader, num_epochs=args.num_epochs, accumulation_steps=args.accumulation_steps, early_stop_patience=args.early_stop_patience, checkpoint_path=f"{save_path}/dsbert_best_model.pth")

    logger.info("=============Evaluation=============")
    
    model = extractor_config.instantiate()
    model.load_state_dict(torch.load(f"{save_path}/dsbert_best_model.pth", map_location=device, weights_only=True))
    trainer = Trainer(model, optimizer=None, device=device)
    evaluate_entity_recognition(trainer, test_dataset)

    logger.info("=============Ending=============")
