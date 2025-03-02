import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from typing import List, Tuple
from dspert.training.evaluation import precision_recall_f1_report
from dspert.models.specific_span_extractor import SpecificSpanExtractor
from dspert.dataset import Dataset

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: SpecificSpanExtractor, optimizer, scheduler=None, grad_clip=None, use_amp=False, device=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda', enabled=use_amp)
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)

    def forward_batch(self, batch: dict) -> torch.Tensor:
        losses = self.model.forward(batch)
        return losses.mean()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        epoch_losses = []

        for batch in tqdm(dataloader, desc="Training"):
            batch['bert_like']['sub_tok_ids'] = batch['bert_like']['sub_tok_ids'].to(self.device)
            batch['bert_like']['sub_mask'] = batch['bert_like']['sub_mask'].to(self.device)
            batch['bert_like']['ori_indexes'] = batch['bert_like']['ori_indexes'].to(self.device)
            batch['seq_lens'] = batch['seq_lens'].to(self.device)
            batch['mask'] = batch['mask'].to(self.device)
            for obj in batch['boundaries_objs']:
                obj['label_ids'] = obj['label_ids'].to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                loss = self.forward_batch(batch)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        logger.info(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    def eval_epoch(self, dataloader: DataLoader) -> Tuple[float, List]:
        self.model.eval()
        epoch_losses = []
        all_pred_chunks = []

        with torch.no_grad():
            for batch in dataloader:
                batch['bert_like']['sub_tok_ids'] = batch['bert_like']['sub_tok_ids'].to(self.device)
                batch['bert_like']['sub_mask'] = batch['bert_like']['sub_mask'].to(self.device)
                batch['bert_like']['ori_indexes'] = batch['bert_like']['ori_indexes'].to(self.device)
                batch['seq_lens'] = batch['seq_lens'].to(self.device)
                batch['mask'] = batch['mask'].to(self.device)
                for obj in batch['boundaries_objs']:
                    obj['label_ids'] = obj['label_ids'].to(self.device)

                with autocast('cuda', enabled=self.use_amp):
                    losses = self.model.forward(batch)
                pred_chunks = self.model.decode(batch)
                
                epoch_losses.append(losses.mean().item())
                all_pred_chunks.extend(pred_chunks)

        avg_loss = np.mean(epoch_losses)
        precision, recall, f1 = precision_recall_f1_report([sample['chunks'] for sample in dataloader.dataset.data], all_pred_chunks)
        logger.info(f"Eval Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return avg_loss, all_pred_chunks

    def predict(self, dataset: Dataset, batch_size: int = 32) -> List:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        self.model.eval()
        all_pred_chunks = []

        with torch.no_grad():
            for batch in dataloader:
                batch['bert_like']['sub_tok_ids'] = batch['bert_like']['sub_tok_ids'].to(self.device)
                batch['bert_like']['sub_mask'] = batch['bert_like']['sub_mask'].to(self.device)
                batch['bert_like']['ori_indexes'] = batch['bert_like']['ori_indexes'].to(self.device)
                batch['seq_lens'] = batch['seq_lens'].to(self.device)
                batch['mask'] = batch['mask'].to(self.device)
                for obj in batch['boundaries_objs']:
                    obj['label_ids'] = obj['label_ids'].to(self.device)

                pred_chunks = self.model.decode(batch)
                all_pred_chunks.extend(pred_chunks)

        return all_pred_chunks

    def train(self, 
              train_dataloader: DataLoader, 
              eval_dataloader: DataLoader = None, 
              num_epochs: int = 10, 
              early_stop_patience: int = 3,
              checkpoint_path: str = "best_model.pth"):
        best_eval_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_dataloader)
            
            if eval_dataloader is not None:
                eval_loss, pred_chunks = self.eval_epoch(eval_dataloader)
                logger.info(f"Predicted chunks: {pred_chunks}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                    logger.info(f"New best eval loss: {best_eval_loss:.4f}, saving checkpoint to {checkpoint_path}")
                    torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                    logger.info(f"No improvement, patience counter: {patience_counter}/{early_stop_patience}")
                
                if patience_counter >= early_stop_patience:
                    logger.info("Early stopping triggered.")
                    break

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(eval_loss)
                    else:
                        self.scheduler.step()
                    logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")