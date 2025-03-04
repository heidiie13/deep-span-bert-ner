import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderConfig:
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'IDENTITY').upper()
        self.in_dim = kwargs.pop('in_dim', None)
        self.hid_dim = kwargs.pop('hid_dim', 128)
        
        self.num_layers = kwargs.pop('num_layers', 1)
        
        self.in_drop_rates = kwargs.pop('in_drop_rates', 0.5)
        self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.5)
        
        if self.arch not in ['IDENTITY', 'FFN', 'LSTM', 'GRU']:
            raise ValueError(f"Invalid encoder architecture {self.arch}")
        
    @property
    def name(self):
        return self.arch
    
    @property
    def out_dim(self):
        if self.arch == 'IDENTITY':
            out_dim = self.in_dim
        else:
            out_dim = self.hid_dim
        return out_dim
    
    def instantiate(self):
        if self.arch == 'IDENTITY':
            return IdentityEncoder(self)
        elif self.arch == 'FFN':
            return FFNEncoder(self)
        elif self.arch in ['LSTM', 'GRU']:
            return RNNEncoder(self)


class Encoder(nn.Module):
    """Encoder từ embeddings sang hidden states."""
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.in_drop_rates)

    def embedded2hidden(self, embedded: torch.Tensor, mask: torch.Tensor = None):
        raise NotImplementedError("Not Implemented `embedded2hidden`")
    
    def forward(self, embedded: torch.Tensor, mask: torch.Tensor = None):
        """Chuyển từ embedded sang hidden states với mask tùy chọn."""
        hidden = self.embedded2hidden(self.dropout(embedded), mask)
        return hidden


class IdentityEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
    
    def embedded2hidden(self, embedded: torch.Tensor, mask: torch.Tensor = None):
        return embedded


class FFNEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.ff_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.in_dim if k == 0 else config.hid_dim, config.hid_dim),
                nn.ReLU(),
                nn.Dropout(config.hid_drop_rate) if config.hid_drop_rate > 0 else nn.Identity()
            ) for k in range(config.num_layers)
        ])
        for block in self.ff_blocks:
            if isinstance(block[0], nn.Linear):
                nn.init.xavier_uniform_(block[0].weight)
                nn.init.zeros_(block[0].bias)

    def embedded2hidden(self, embedded: torch.Tensor, mask: torch.Tensor = None):
        hidden = embedded
        for ff_block in self.ff_blocks:
            hidden = ff_block(hidden)
        return hidden


class RNNEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        rnn_class = nn.LSTM if config.arch == 'LSTM' else nn.GRU
        self.rnn = rnn_class(
            input_size=config.in_dim,
            hidden_size=config.hid_dim // 2,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.hid_drop_rate if config.num_layers > 1 else 0.0
        )
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def embedded2hidden(self, embedded: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            lengths = mask.sum(dim=1).long()
            lengths = lengths.clamp(max=embedded.size(1))
            packed = pack_padded_sequence(
                embedded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            rnn_outs, _ = self.rnn(packed)
            rnn_outs, _ = pad_packed_sequence(rnn_outs, batch_first=True, total_length=embedded.size(1))
        else:
            rnn_outs, _ = self.rnn(embedded)
        return rnn_outs