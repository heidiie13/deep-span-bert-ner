from typing import List
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from torch import nn

from dsbert.nn_modules.aggregation import SequenceGroupAggregating
from dsbert.utils import seq_lens2mask

class BertLikeConfig:
    def __init__(self, **kwargs):
        self.tokenizer: PreTrainedTokenizer = kwargs.pop('tokenizer')
        self.bert_like: PreTrainedModel = kwargs.pop('bert_like')
        self.hid_dim = self.bert_like.config.hidden_size
        self.num_layers = self.bert_like.config.num_hidden_layers
        
        self.arch = kwargs.pop('arch', 'BERT')
        self.freeze = kwargs.pop('freeze', True)
        self.group_agg_mode = kwargs.pop('group_agg_mode', 'mean')
        
    @property
    def name(self):
        return self.arch

    @property
    def out_dim(self):
        return self.hid_dim
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['bert_like'] = None
        return state

    def exemplify(self, entry: dict):
        """
        Converts a token sequence into sub-token IDs and original token indexes.
        Limits each token to 3 sub-tokens if the sequence exceeds max length.

        Args:
            entry (dict): A dictionary containing a 'tokens' key with a list of tokens.

        Returns:
            dict: A dictionary containing:
                - 'sub_tok_ids': Tensor of sub-token IDs including special tokens (CLS and SEP).
                - 'ori_indexes': Tensor of original token indexes for each sub-token.
        """

        tokens = entry['tokens']
        sub_tokens_nested = [self.tokenizer.tokenize(token) for token in tokens]
        sub_tokens = [sub_tok for sublist in sub_tokens_nested for sub_tok in sublist]

        # Check if length exceeds maximum allowed length
        if len(sub_tokens) > self.tokenizer.model_max_length - 2:
            # If exceeds, limit each token to 3 sub-tokens
            sub_tokens_nested = [self.tokenizer.tokenize(token)[:3] for token in tokens]
            sub_tokens = [sub_tok for sublist in sub_tokens_nested for sub_tok in sublist]
            
        ori_indexes = [i for i, sublist in enumerate(sub_tokens_nested) for _ in sublist]

        sub_tok_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(sub_tokens) + [self.tokenizer.sep_token_id]

        example = {
            'sub_tok_ids': torch.tensor(sub_tok_ids),
            'ori_indexes': torch.tensor(ori_indexes)
        }
        return example

    def batchify(self, batch_ex: List[dict]):
        """
        Constructs a batch from a list of examples by processing sub-token IDs and original indexes.

        Args:
            batch_ex (List[dict]): A list of dictionaries, each containing:
                - 'sub_tok_ids': Tensor of sub-token IDs.
                - 'ori_indexes': Tensor of original token indexes for each sub-token.

        Returns:
            dict: A dictionary containing:
                - 'sub_tok_ids': Padded tensor of sub-token IDs across the batch.
                - 'sub_mask': Mask tensor indicating valid sub-token positions.
                - 'ori_indexes': Padded tensor of original token indexes.
        """

        batch_sub_tok_ids = [ex['sub_tok_ids'] for ex in batch_ex]
        sub_tok_seq_lens = torch.tensor([ids.size(0) for ids in batch_sub_tok_ids])
        batch_sub_mask = seq_lens2mask(sub_tok_seq_lens)
        batch_sub_tok_ids = torch.nn.utils.rnn.pad_sequence(
            batch_sub_tok_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        batch_ori_indexes = [ex['ori_indexes'] for ex in batch_ex]
        batch_ori_indexes = torch.nn.utils.rnn.pad_sequence(
            batch_ori_indexes,
            batch_first=True,
            padding_value=-1
        )
        
        return {
            'sub_tok_ids': batch_sub_tok_ids,
            'sub_mask': batch_sub_mask,
            'ori_indexes': batch_ori_indexes
        }

    def instantiate(self):
        return BertLikeEmbedder(self)


class BertLikeEmbedder(nn.Module):
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.bert_like: PreTrainedModel = config.bert_like
        self.freeze = config.freeze
        self.group_agg = SequenceGroupAggregating(mode=config.group_agg_mode)

    @property
    def freeze(self):
        return self._freeze

    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.bert_like.requires_grad_(not freeze)

    def forward(self, sub_tok_ids, sub_mask, ori_indexes=None):
        bert_outs = self.bert_like(
            input_ids=sub_tok_ids,
            attention_mask=sub_mask.long(),
            output_hidden_states=True
        )

        bert_hidden = bert_outs.last_hidden_state[:, 1:-1]
        sub_mask = sub_mask[:, 2:]
        bert_hidden = self.group_agg(bert_hidden, ori_indexes)
        
        all_bert_hidden = [self.group_agg(h[:, 1:-1], ori_indexes) for h in bert_outs.hidden_states]
        return bert_hidden, all_bert_hidden