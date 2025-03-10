import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from dsbert.training.evaluation import precision_recall_f1_report
from dsbert.models.deep_span_extractor import DeepSpanExtractor
from dsbert.dataset import Dataset

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, 
                 model: DeepSpanExtractor, 
                 optimizer: Optional[torch.optim.Optimizer] = None, 
                 scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None, 
                 grad_clip: Optional[float] = None, 
                 use_amp: bool = False, 
                 device: Optional[torch.device] = None):
        """Initialize the Trainer."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.use_amp = use_amp and device.type == 'cuda'

        self.scaler = GradScaler(enabled=use_amp)
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)

    def _to_device(self, batch: Dict) -> Dict:
        batch['bert_like']['sub_tok_ids'] = batch['bert_like']['sub_tok_ids'].to(self.device)
        batch['bert_like']['sub_mask'] = batch['bert_like']['sub_mask'].to(self.device)
        batch['bert_like']['ori_indexes'] = batch['bert_like']['ori_indexes'].to(self.device)
        batch['seq_lens'] = batch['seq_lens'].to(self.device)
        batch['mask'] = batch['mask'].to(self.device)
        for obj in batch['boundaries_objs']:
            obj['label_ids'] = obj['label_ids'].to(self.device)
        return batch

    def forward_batch(self, batch: Dict) -> torch.Tensor:
        losses,_ = self.model.forward(batch)
        return losses.mean()

    def train_epoch(self, dataloader: DataLoader, accumulation_steps: int = 1) -> float:
        if self.optimizer is None:
            raise ValueError("Optimizer is not defined. Please provide an optimizer to the Trainer.")
        
        self.model.train()
        epoch_losses = []
        self.optimizer.zero_grad()

        for i, batch in enumerate(tqdm(dataloader, desc="Training")):
            batch = self._to_device(batch)
            with autocast(self.device.type, enabled=self.use_amp):
                batch_loss = self.forward_batch(batch)
                loss = batch_loss / accumulation_steps
                
            self.scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                if self.grad_clip is not None and self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()

                if (scale_after == scale_before) and self.scheduler is not None: # Fix: raise warning when using GradScaler with scheduler
                        self.scheduler.step()
                self.optimizer.zero_grad()

                epoch_losses.append(batch_loss.item())

        avg_loss = np.mean(epoch_losses)
        logger.info(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    def eval_epoch(self, dataloader: DataLoader) -> Tuple[float, List]:
        self.model.eval()
        epoch_losses = []
        all_pred_chunks = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = self._to_device(batch)
                with autocast('cuda', enabled=self.use_amp):
                    losses, states = self.model.forward(batch)
                pred_chunks = self.model.decode(batch, **states)

                epoch_losses.append(losses.mean().item())
                all_pred_chunks.extend(pred_chunks)

        avg_loss = np.mean(epoch_losses)
        true_chunks = [sample['chunks'] for sample in dataloader.dataset.data]
        precision, recall, f1, *_ = precision_recall_f1_report(true_chunks, all_pred_chunks)
        logger.info(f"Eval Loss: {avg_loss:.4f}, (Micro) Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return avg_loss, all_pred_chunks, f1

    def predict(self, dataset: Dataset, batch_size: int = 32) -> List:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        self.model.eval()
        all_pred_chunks = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = self._to_device(batch)
                pred_chunks = self.model.decode(batch)
                all_pred_chunks.extend(pred_chunks)

        return all_pred_chunks

    def train(self, 
            train_dataloader: DataLoader, 
            eval_dataloader: Optional[DataLoader] = None, 
            num_epochs: int = 10, 
            early_stop_patience: Optional[int] = None, 
            checkpoint_path: str = "best_model.pth",
            accumulation_steps: int = 1,
            save_by_loss: bool = False):
        
        if early_stop_patience is None:
            early_stop_patience = num_epochs
        elif early_stop_patience <= 0:
            raise ValueError("early_stop_patience must be a positive integer or None.")
            
        best_eval_loss = float('inf')
        best_eval_f1 = -float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_dataloader, accumulation_steps)

            if eval_dataloader is not None:
                eval_loss, pred_chunks, f1 = self.eval_epoch(eval_dataloader)

                if save_by_loss:
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        patience_counter = 0
                        torch.save(self.model.state_dict(), checkpoint_path)
                        logger.info(f"New best eval loss: {eval_loss:.4f}, saving checkpoint to {checkpoint_path}")
                    else:
                        patience_counter += 1
                else:
                    if f1 > best_eval_f1:
                        best_eval_f1 = f1
                        patience_counter = 0
                        torch.save(self.model.state_dict(), checkpoint_path)
                        logger.info(f"New best F1-score: {f1:.4f}, saving checkpoint to {checkpoint_path}")
                    else:
                        patience_counter += 1

                logger.info(f"Patience counter: {patience_counter}/{early_stop_patience}")

                if patience_counter >= early_stop_patience:
                    logger.info("Early stopping triggered.")
                    break

            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"Finetune learning rate: {self.optimizer.param_groups[1]['lr']:.6f}")
