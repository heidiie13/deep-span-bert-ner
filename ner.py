from dspert.models.specific_span_extractor import SpecificSpanExtractorConfig
from dspert.models.decoders.specific_span_classification import SpecificSpanClsDecoderConfig
from dspert.models.encoders.bert_like import BertLikeConfig
from dspert.models.encoders.span_bert_like import SpanBertLikeConfig
from dspert.training.trainer import Trainer
from dspert.training.evaluation import evaluate_entity_recognition
from dspert.dataset import Dataset
from dspert.utils import load_data, load_pretrained, count_parameters
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    
    handlers=[
        logging.FileHandler(f"log/training_phoner.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("=============Starting=============")

name_data = 'phoner_covid19'

train_data, dev_data, test_data = load_data(name_data)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")

bert_model, tokenizer = load_pretrained('bert-base-uncased')
bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert_model, freeze=False)
span_bert_like_config = SpanBertLikeConfig(bert_like=bert_model, freeze=False, share_weights_ext=False, share_weights_int=True)
decoder_config = SpecificSpanClsDecoderConfig()
extractor_config = SpecificSpanExtractorConfig(
    decoder=decoder_config,
    bert_like=bert_like_config,
    span_bert_like=span_bert_like_config,
    share_interm2=False,
    max_span_size=10
)

train_dataset = Dataset(train_data, extractor_config)
dev_dataset = Dataset(dev_data, extractor_config)
test_dataset = Dataset(test_data, extractor_config)

train_dataset.build_vocabs(dev_data, test_data)

logger.info(f"\nSummary train dataset: {train_dataset.summary}")

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle= True, collate_fn=train_dataset.collate)
dev_dataloader = DataLoader(dev_dataset, batch_size=16, shuffle= False, collate_fn=dev_dataset.collate)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle= False, collate_fn=dev_dataset.collate)

model = extractor_config.instantiate()
num_params, trainable_params = count_parameters(model)
logger.info(f"Number of parameters: {num_params}, trainable parameters: {trainable_params}")

# optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
# trainer = Trainer(model, optimizer, scheduler=scheduler, grad_clip=5.0, use_amp=True, device=device)

# trainer.train(test_dataloader, test_dataloader, num_epochs=1, early_stop_patience=5, checkpoint_path=f"{name_data}_best_model.pth")

# logger.info("=============Evaluation=============")

# evaluate_entity_recognition(trainer, test_dataset)

# loaded_model = extractor_config.instantiate()
# loaded_model.load_state_dict(torch.load("best_model.pth", weights_only=True))
# loaded_trainer = Trainer(loaded_model, optimizer=None, device=device)
# pred_chunks = loaded_trainer.predict(test_dataset)
# print("Predicted chunks from loaded model:", pred_chunks)

# logger.info("=============Ending=============")
