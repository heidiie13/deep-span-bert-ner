# -*- coding: utf-8 -*-
from typing import Dict, List
from collections import Counter
import logging
import torch
import torch.nn as nn

from dspert.models.encoders.encode import EncoderConfig

logger = logging.getLogger(__name__)

class SpecificSpanClsDecoderConfig:
    def __init__(self, **kwargs):
        # Cấu hình mạng affine (FFN đơn giản)
        self.affine = kwargs.pop('affine', EncoderConfig(arch='FFN', hid_dim=300, num_layers=1, hid_drop_rate=0.2))
        
        # Giới hạn kích thước span tối đa
        self.max_span_size_ceiling = kwargs.pop('max_span_size_ceiling', 25)
        self.max_span_size = kwargs.pop('max_span_size', None)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.2)
        self.criterion = kwargs.pop('criterion', 'CrossEntropy')
        
        # Nhãn và từ vựng
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        
    @property
    def name(self):
        return f"{self.affine.arch}_{self.criterion}"

    @property
    def in_dim(self):
        return self.affine.in_dim - self.size_emb_dim
        
    @in_dim.setter
    def in_dim(self, dim: int):
        if dim is not None:
            self.affine.in_dim = dim + self.size_emb_dim

    def instantiate_criterion(self, **kwargs):
        if self.criterion == 'LeakyReLU':
            return torch.nn.LeakyReLU(**kwargs)
        else:
            return torch.nn.CrossEntropyLoss(**kwargs)
    
    def exemplify(self, entry: Dict) -> Dict:
        """
        Chuyển đổi một entry thành định dạng phù hợp với decoder.
        - entry: Dictionary chứa 'tokens' và 'chunks' (danh sách (label, start, end)).
        - Trả về: Dictionary chứa 'boundaries_obj' với 'label_ids'.
        """
        seq_len = len(entry['tokens'])
        total_spans = sum(seq_len - k + 1 for k in range(1, min(self.max_span_size, seq_len) + 1))
        
        # Khởi tạo label_ids với nhãn mặc định là none_label
        label_ids = torch.full((total_spans,), self.idx2label.index(self.none_label), dtype=torch.long)
        
        # Tạo danh sách spans tương ứng
        spans = [(start, start + k) for k in range(1, min(self.max_span_size, seq_len) + 1) 
                 for start in range(seq_len - k + 1)]
        
        for label, start, end in entry.get('chunks', []):
            if end - start <= self.max_span_size:
                span_idx = spans.index((start, end))
                if label in self.idx2label:
                    label_ids[span_idx] = self.idx2label.index(label)
                else:
                    logger.warning(f"Label '{label}' not in idx2label, ignoring this chunk.")
        
        return {'boundaries_obj': {'label_ids': label_ids}}

    def batchify(self, batch_examples: List[Dict]) -> Dict:
        """
        Tạo batch từ danh sách examples.
        - batch_examples: List các dictionary từ exemplify.
        - Trả về: Dictionary chứa 'boundaries_objs'. 'seq_lens' đã được tạo trong Dataset.collate.
        """
        batch = {'boundaries_objs': [ex['boundaries_obj'] for ex in batch_examples]}
        return batch
    
    def build_vocab(self, *partitions):
        if self.idx2label is None:
            counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
            self.idx2label = [self.none_label] + list(counter.keys())
        else:
            self.idx2label = [self.none_label] + self.idx2label
        logger.info(f"Labels: {self.idx2label}")
        
        # Calculate `max_span_size` according to data
        span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
        logger.info(f"Max span size in data: {max(span_sizes)}")
        
        if (self.max_span_size is not None) and (self.max_span_size > self.max_span_size_ceiling):
            raise ValueError("Max span size cannot be greater than max span size ceiling")
        
        if self.max_span_size is None:
            self.max_span_size = min(max(span_sizes), self.max_span_size_ceiling)
            logger.warning(f"The `max_span_size` is set to {self.max_span_size}")
        
        size_counter = Counter(end-start for data in partitions for entry in data for label, start, end in entry['chunks'])
        
        num_spans = sum(size_counter.values())
        num_oov_spans = sum(num for size, num in size_counter.items() if size > self.max_span_size)
        if num_oov_spans > 0:
            logger.warning(f"OOV positive spans: {num_oov_spans} ({num_oov_spans/num_spans*100:.2f}%) with max_span_size: {self.max_span_size}")
            
    def instantiate(self):
        return SpecificSpanClsDecoder(self)

class SpecificSpanClsDecoder(nn.Module):
    def __init__(self, config: SpecificSpanClsDecoderConfig):
        super().__init__()
        
        self.max_span_size = config.max_span_size
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.label2idx = {label: i for i, label in enumerate(self.idx2label)}
        
        self.affine = config.affine.instantiate()
        if config.size_emb_dim > 0:
            self.size_embedding = nn.Embedding(self.max_span_size + 1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        self.dropout = nn.Dropout(config.hid_drop_rate)
        self.hid2logit = nn.Linear(config.affine.out_dim, len(self.idx2label))
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')

    def get_logits(self, batch: Dict, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        """
        Tính toán logits từ full_hidden và all_query_hidden.
        Đầu vào:
        - full_hidden: [batch_size, seq_len, hid_dim]
        - all_query_hidden: {k: [batch_size, seq_len - k + 1, hid_dim]}
        """
        batch_logits = []
        for i, curr_len in enumerate(batch["seq_lens"].cpu().tolist()):
            # Thu thập hidden states cho tất cả các span từ 1 đến max_span_size
            span_hidden = []
            for k in range(1, min(self.max_span_size, curr_len) + 1):
                if k == 1:
                    # Span kích thước 1: sử dụng full_hidden
                    span_hidden.append(full_hidden[i, :curr_len, :])
                else:   
                    # Span kích thước k > 1: sử dụng all_query_hidden
                    span_hidden.append(all_query_hidden[k][i, :curr_len - k + 1, :])
            span_hidden = torch.cat(span_hidden, dim=0)  # [total_spans, hid_dim]

            # Thêm size embedding nếu có
            if hasattr(self, 'size_embedding'):
                size_ids = torch.cat([torch.full((curr_len - k + 1,), k, dtype=torch.long, device=span_hidden.device) 
                                     for k in range(1, min(self.max_span_size, curr_len) + 1)], dim=0)
                size_embedded = self.size_embedding(size_ids)
                span_hidden = torch.cat([span_hidden, size_embedded], dim=-1)
            
            # Tính logits
            affined = self.affine(span_hidden)
            logits = self.hid2logit(self.dropout(affined))
            batch_logits.append(logits)
        
        return batch_logits

    def forward(self, batch: Dict, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        """
        Tính toán loss từ batch.
        """
        batch_logits = self.get_logits(batch, full_hidden, all_query_hidden)
        
        losses = []
        for logits, boundaries_obj in zip(batch_logits, batch["boundaries_objs"]):
            label_ids = boundaries_obj["label_ids"]
            loss = self.criterion(logits, label_ids)
            losses.append(loss)
        
        return torch.stack(losses)

    def decode(self, batch: Dict, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        """
        Giải mã để lấy danh sách các chunk (label, start, end).
        """
        batch_logits = self.get_logits(batch, full_hidden, all_query_hidden)
        
        batch_chunks = []
        for logits, curr_len in zip(batch_logits, batch["seq_lens"].cpu().tolist()):
            confidences, label_ids = logits.softmax(dim=-1).max(dim=-1)
            labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]

            spans = []
            for k in range(1, min(self.max_span_size, curr_len) + 1):
                for start in range(curr_len - k + 1):
                    spans.append((start, start + k))
            chunks = [(label, start, end) for label, (start, end) in zip(labels, spans) if label != self.none_label]
            batch_chunks.append(chunks)
        
        return batch_chunks
    
    from typing import List, Tuple
    
    
def reinit_embedding_(embedding: nn.Embedding):
    """Reinitialize an embedding layer with a Gaussian distribution."""
    nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
    if embedding.padding_idx is not None:
        embedding.weight.data[embedding.padding_idx].zero_()

def reinit_layer_(layer: nn.Module, activation: str = 'relu'):
    """Reinitialize a linear layer using Xavier/Glorot initialization."""
    if isinstance(layer, nn.Linear):
        if activation.lower() == 'sigmoid':
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('sigmoid'))
        else:
            nn.init.kaiming_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(layer.bias)