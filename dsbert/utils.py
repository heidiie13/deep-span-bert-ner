import torch
from transformers import AutoModel, AutoTokenizer
import json
import os
from typing import List, Dict, Tuple


def seq_lens2mask(seq_lens: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    Create a mask from a list of sequence lengths.
    Args:
        seq_lens: A tensor containing the length of each sequence (shape: [batch_size])
        max_len: The maximum length of sequences (if None, use the max of seq_lens)
    Returns:
        mask: A binary tensor (shape: [batch_size, max_len]), True for valid positions
    """
    batch_size = seq_lens.size(0)
    if max_len is None:
        max_len = seq_lens.max().item()
    
    indices = torch.arange(max_len, device=seq_lens.device).unsqueeze(0).expand(batch_size, max_len)
    mask = indices < seq_lens.unsqueeze(1)
    return mask

def count_parameters(model):
    """
    Count the total and trainable parameters in a model.

    Args:
        model: A PyTorch model instance.

    Returns:
        A tuple containing:
        - total_params: The total number of parameters in the model.
        - trainable_params: The number of parameters that require gradients (i.e., are trainable).
    """

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def load_pretrained(pretrained_str):
    """
    Load a pre-trained model and its corresponding tokenizer.

    Args:
        pretrained_str: The name of the pre-trained model, e.g. "bert-base-uncased" or "vinai/phobert-base"

    Returns:
        A tuple of (pretrained_model, tokenizer), where pretrained_model is an instance of AutoModel and tokenizer is an instance of AutoTokenizer
    """
    model_name = pretrained_str
    model_path = f"assets/transformers/{model_name}"

    pretrained_model = AutoModel.from_pretrained(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True)

    return pretrained_model, tokenizer


def read_json(file_path: str) -> List:
    """
    Read a JSON file and convert 'chunks' entries from lists to tuples.

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of dictionaries from the JSON file with 'chunks' entries as tuples.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not exists.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for sample in data:
        sample['chunks'] = [tuple(chunk) for chunk in sample['chunks']]
        
    return data


def load_data(data_name: str):
    """
    Load processed dataset splits from JSON files.

    Args:
        data_name: The name of the dataset directory within 'data/processed_data'.

    Returns:
        A tuple containing three lists:
        - train_data: A list of training samples loaded from 'train.json'.
        - dev_data: A list of development samples loaded from 'dev.json'.
        - test_data: A list of test samples loaded from 'test.json'.
    
    Raises:
        FileNotFoundError: If any of the expected JSON files do not exist.
    """

    base_path = f"data/processed_data/{data_name}"
    
    file_names = {
        'train': os.path.join(base_path, 'train.json'),
        'dev': os.path.join(base_path, 'dev.json'),
        'test': os.path.join(base_path, 'test.json'),
    }
    
    train_data = read_json(file_names['train'])
    dev_data = read_json(file_names['dev'])
    test_data = read_json(file_names['test'])
    
    return train_data, dev_data, test_data