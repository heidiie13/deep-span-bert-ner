from typing import List
import torch

from dsbert.models.deep_span_extractor import DeepSpanExtractorConfig
from dsbert.utils import seq_lens2mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], config: DeepSpanExtractorConfig, training: bool=True):
        """Construct a dataset from a list of data entries.

        Args:
            data (List[dict]): A list of data entries. Each entry is a dictionary
                with the following keys:
                    'tokens': TokenSequence
                    'chunks': List[tuple(chunk_type, chunk_start, chunk_end)]
                    'relations': List[tuple]

        Returns:
            Dataset: A dataset object.
        """
        super().__init__()
        self.data = data
        self.config = config
        self.training = training
        
    def __len__(self):
        return len(self.data)
        
    @property
    def summary(self):
        """Returns a summary of the dataset.

        The summary includes the following information:
            - The number of sequences in the dataset.
            - The average and maximum lengths of the `tokens` sequences.
            - The total number of chunks and the number of chunk types.

        Returns:
            str: A string containing the summary information.
        """
        summary = []
        num_seqs = len(self.data)
        summary.append(f"The dataset consists {num_seqs:,} sequences")
        
        if 'tokens' in self.data[0]:
            seq_lens = [len(entry['tokens']) for entry in self.data]
            ave_len, max_len = sum(seq_lens)/len(seq_lens), max(seq_lens)
            summary.extend([f"The average `tokens` length is {ave_len:,.1f}", 
                            f"The maximum `tokens` length is {max_len:,}"])
        
        if 'chunks' in self.data[0]:
            num_chunks = sum(len(entry['chunks']) for entry in self.data)
            num_chunk_types = len({ck[0] for entry in self.data for ck in entry['chunks']})
            summary.append(f"The dataset has {num_chunks:,} chunks of {num_chunk_types:,} types")
        
        return "\n".join(summary)
        
        
    def build_vocabs(self, *others):
        self.config.build_vocabs(self.data, *others)
        
    def __getitem__(self, i):
        entry = self.data[i]
        example = {'tokens': entry['tokens']}
        example.update(self.config.exemplify(entry))
        return example
        
    def collate(self, batch_examples: List[dict]):
        batch = {}
        batch['tokens'] = [ex['tokens'] for ex in batch_examples]
        batch['seq_lens'] = torch.tensor([len(tokenized_text) for tokenized_text in batch['tokens']])        
        batch['mask'] = seq_lens2mask(batch['seq_lens'])
        
        batch.update(self.config.batchify(batch_examples))
        return batch