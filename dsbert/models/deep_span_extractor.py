from typing import List, Dict, Union
import torch.nn as nn
from dsbert.models.encoders.encode import EncoderConfig
from dsbert.models.deep_span_classification import DeepSpanClsDecoderConfig
from dsbert.models.encoders.bert_like import BertLikeConfig
from dsbert.models.encoders.span_bert_like import SpanBertLikeConfig
import torch

class DeepSpanExtractorConfig:
    def __init__(self, decoder: DeepSpanClsDecoderConfig, **kwargs):
        self.bert_like: BertLikeConfig = kwargs.pop('bert_like', None)
        self.span_bert_like: SpanBertLikeConfig = kwargs.pop('span_bert_like', None)
        self.intermediate2: EncoderConfig = kwargs.pop('intermediate2', EncoderConfig(arch='LSTM', hid_dim=400))
        self.share_interm2 = kwargs.pop('share_interm2', True)
        self.decoder = decoder
        self.max_span_size = kwargs.pop('max_span_size', None)
        
        if self.intermediate2 is not None:
            self.intermediate2.in_dim = self.bert_like.out_dim
            self.decoder.in_dim = self.intermediate2.out_dim
        else:
            self.decoder.in_dim = self.bert_like.out_dim      
        
        if self.max_span_size is not None:  
            self.decoder.max_span_size = self.max_span_size
            self.span_bert_like.max_span_size = self.max_span_size
    
    @property
    def name(self):
        return self.span_bert_like.name + f"({self.decoder.name})"
    def build_vocabs(self, *partitions):
        self.decoder.build_vocab(*partitions)
        self.max_span_size = self.decoder.max_span_size
        self.span_bert_like.max_span_size = self.decoder.max_span_size
        
    @property
    def span_intermediate2(self):
        if self.share_interm2:
            return None
        elif self.span_bert_like.share_weights_int:
            return self.intermediate2
        else:
            return ConfigList([self.intermediate2 for k in range(2, self.span_bert_like.max_span_size+1)])
        
    def exemplify(self, entry: Dict) -> Dict:
        example = {}
        example['bert_like'] = self.bert_like.exemplify(entry)
        example.update(self.decoder.exemplify(entry))
        return example
    
    def batchify(self, batch_examples: List[Dict]) -> Dict:
        batch = {}
        batch['bert_like'] = self.bert_like.batchify([ex['bert_like'] for ex in batch_examples])
        batch.update(self.decoder.batchify(batch_examples))
        return batch
    
    def instantiate(self):
        return DeepSpanExtractor(self)


class DeepSpanExtractor(nn.Module):
    def __init__(self, config: DeepSpanExtractorConfig):
        super().__init__()
        
        self.bert_like = config.bert_like.instantiate()
        self.span_bert_like = config.span_bert_like.instantiate()
        self.intermediate2 = config.intermediate2.instantiate() if config.intermediate2 else None
        self.span_intermediate2 = config.span_intermediate2.instantiate() if config.span_intermediate2 else None
        self.decoder = config.decoder.instantiate()
        self.max_span_size = config.max_span_size
        
    def forward2states(self, batch: Dict) -> Dict:
        bert_hidden, all_bert_hidden = self.bert_like(batch['bert_like']['sub_tok_ids'], 
                                                    batch['bert_like']['sub_mask'], 
                                                    batch['bert_like']['ori_indexes'])
        
        all_last_query_states = self.span_bert_like(all_bert_hidden)
        
        if self.intermediate2 is not None:
            bert_hidden = self.intermediate2(bert_hidden, batch['mask'])
            
            new_all_last_query_states = {}
            for k, query_hidden in all_last_query_states.items():
                num_spans = query_hidden.size(1)
                curr_mask = torch.zeros(query_hidden.size(0), num_spans, dtype=torch.bool, device=batch['seq_lens'].device)
                for i in range(num_spans):
                    span_mask = batch['mask'][:, i:i+k]
                    curr_mask[:, i] = span_mask.sum(dim=1) > 0

                if self.span_intermediate2 is None:
                    new_all_last_query_states[k] = self.intermediate2(query_hidden, curr_mask)
                elif not isinstance(self.span_intermediate2, nn.ModuleList):
                    new_all_last_query_states[k] = self.span_intermediate2(query_hidden, curr_mask)
                else:
                    new_all_last_query_states[k] = self.span_intermediate2[k-2](query_hidden, curr_mask)
            all_last_query_states = new_all_last_query_states
        
        return {'full_hidden': bert_hidden, 'all_query_hidden': all_last_query_states}
    
    def forward(self, batch: Dict, return_states: bool = True) -> Union[Dict, tuple]:
        states = self.forward2states(batch)
        losses = self.decoder(batch, **states)
        
        if return_states:
            return losses, states
        return losses
    
    def decode(self, batch: Dict, **states) -> Dict:
        if len(states) == 0:
            states = self.forward2states(batch)
        return self.decoder.decode(batch, **states)

class ConfigList():
    def __init__(self, config_list: List=None):

        self.config_list = config_list
        
    def instantiate(self):
        return nn.ModuleList([c.instantiate() for c in self.config_list])