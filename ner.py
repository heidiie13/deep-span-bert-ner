import argparse
import logging
import os
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from dsbert.models.deep_span_extractor import SpecificSpanExtractorConfig
from dsbert.models.deep_span_classification import DeepSpanClsDecoderConfig
from dsbert.models.encoders.bert_like import BertLikeConfig
from dsbert.models.encoders.span_bert_like import SpanBertLikeConfig
from dsbert.training.trainer import Trainer
from dsbert.training.evaluation import evaluate_entity_recognition
from dsbert.dataset import Dataset
from dsbert.utils import load_data, load_pretrained, count_parameters

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
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="grad_clip")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--use_amp", default=False, action="store_true", help="use_amp")
    parser.add_argument("--early_stop_patience", type=int, default=3, help="early_stop_patience")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="accumulation_steps")
    
    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_to_args(parser)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path = f"cache/{args.dataset}/{timestamp}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
            
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        
        handlers=[
            logging.FileHandler(f"{save_path}/training.log"),
            logging.StreamHandler()
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
    extractor_config = SpecificSpanExtractorConfig(
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

    model = extractor_config.instantiate().to(device)
    num_params, trainable_params = count_parameters(model)
    logger.info(f"Number of parameters: {num_params}, trainable parameters: {trainable_params}")

    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    num_training_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(0.2 * num_training_steps)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    trainer = Trainer(model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip, use_amp=True, device=device)

    torch.save(extractor_config, f"{save_path}/dsbert_config.pth")
    trainer.train(train_dataloader, dev_dataloader, num_epochs=args.num_epochs, accumulation_steps=args.accumulation_steps, early_stop_patience=args.early_stop_patience, checkpoint_path=f"{save_path}/dsbert_best_model.pth")

    logger.info("=============Evaluation=============")

    evaluate_entity_recognition(trainer, test_dataset)

    logger.info("=============Ending=============")
