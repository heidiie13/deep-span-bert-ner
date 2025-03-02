import torch
from transformers import AutoModel, AutoTokenizer
import json
import os
from typing import List, Dict, Tuple


def seq_lens2mask(seq_lens: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    Tạo mask từ danh sách độ dài chuỗi.
    Args:
        seq_lens: Tensor chứa độ dài của từng chuỗi (shape: [batch_size])
        max_len: Độ dài tối đa của chuỗi (nếu None, lấy max của seq_lens)
    Returns:
        mask: Tensor nhị phân (shape: [batch_size, max_len]), True cho vị trí hợp lệ
    """
    batch_size = seq_lens.size(0)
    if max_len is None:
        max_len = seq_lens.max().item()
    
    indices = torch.arange(max_len, device=seq_lens.device).unsqueeze(0).expand(batch_size, max_len)
    mask = indices < seq_lens.unsqueeze(1)
    return mask

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def load_pretrained(pretrained_str):
    model_name = pretrained_str
    model_path = f"assets/transformers/{model_name}"

    # Load pre-trained model
    pretrained_model = AutoModel.from_pretrained(model_path)

    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return pretrained_model, tokenizer


def read_json(file_path: str) -> List:
    """
    Đọc dữ liệu từ file JSON và trả về một list.

    Parameters
    ----------
    file_path : str
        Đường dẫn đến file JSON.

    Returns
    -------
    List
        Danh sách dữ liệu từ file JSON (có thể là List[Dict] hoặc List[str]).

    Raises
    ------
    FileNotFoundError
        Nếu file_path không tồn tại.
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
    Tải dữ liệu từ thư mục data/processed_data/{data_name} sử dụng hàm read_json.

    Parameters
    ----------
    data_name : str
        Tên thư mục chứa dữ liệu (ví dụ: 'my_dataset').

    Returns
    -------
    Tuple
        - train_data: Danh sách các mẫu huấn luyện.
        - dev_data: Danh sách các mẫu phát triển (validation).
        - test_data: Danh sách các mẫu kiểm tra.
    """
    # Đường dẫn thư mục dữ liệu
    base_path = f"data/processed_data/{data_name}"
    
    # Các file cần tải
    file_names = {
        'train': os.path.join(base_path, 'train.json'),
        'dev': os.path.join(base_path, 'dev.json'),
        'test': os.path.join(base_path, 'test.json'),
    }
    
    # Tải dữ liệu bằng hàm read_json
    train_data = read_json(file_names['train'])
    dev_data = read_json(file_names['dev'])
    test_data = read_json(file_names['test'])
    
    return train_data, dev_data, test_data
    
    
    
# FLAT = 0       # Flat entities
# NESTED = 1     # Nested entities
# ARBITRARY = 2  # Arbitrarily overlapping entities


# def _is_overlapping(chunk1: tuple, chunk2: tuple):
#     # `NESTED` or `ARBITRARY`
#     (_, s1, e1), (_, s2, e2) = chunk1, chunk2
#     return (s1 < e2 and s2 < e1)


# def _is_ordered_nested(chunk1: tuple, chunk2: tuple):
#     # `chunk1` is nested in `chunk2`
#     (_, s1, e1), (_, s2, e2) = chunk1, chunk2
#     return (s2 <= s1 and e1 <= e2)


# def _is_nested(chunk1: tuple, chunk2: tuple):
#     # `NESTED`
#     (_, s1, e1), (_, s2, e2) = chunk1, chunk2
#     return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


# def _is_clashed(chunk1: tuple, chunk2: tuple, allow_level: int=NESTED):
#     if allow_level == FLAT:
#         return _is_overlapping(chunk1, chunk2)
#     elif allow_level == NESTED:
#         return _is_overlapping(chunk1, chunk2) and not _is_nested(chunk1, chunk2)
#     else:
#         return False


# def filter_clashed_by_priority(chunks: List[tuple], allow_level: int=NESTED):
#     filtered_chunks = []
#     for ck in chunks:
#         if all(not _is_clashed(ck, ex_ck, allow_level=allow_level) for ex_ck in filtered_chunks):
#             filtered_chunks.append(ck)
#     return filtered_chunks


# def detect_overlapping_level(chunks: List[tuple]):
#     level = FLAT
#     for i, ck1 in enumerate(chunks):
#         for ck2 in chunks[i+1:]:
#             if _is_nested(ck1, ck2):
#                 level = NESTED
#             elif _is_overlapping(ck1, ck2):
#                 # Non-nested overlapping -> `ARBITRARY`
#                 return ARBITRARY
#     return level


# def detect_nested(chunks1: List[tuple], chunks2: List[tuple]=None, strict: bool=True):
#     """Return chunks from `chunks1` that are nested in any chunk from `chunks2`. 
#     """
#     if chunks2 is None:
#         chunks2 = chunks1
    
#     nested_chunks = []
#     for ck1 in chunks1:
#         if any(_is_ordered_nested(ck1, ck2) and (ck1 != ck2) and (not strict or ck1[1:] != ck2[1:]) for ck2 in chunks2):
#             nested_chunks.append(ck1)
#     return nested_chunks

# def count_nested(chunks1: List[tuple], chunks2: List[tuple]=None, strict: bool=True):
#     return len(detect_nested(chunks1, chunks2=chunks2, strict=strict))


# def chunk_pair_distance(chunk1: tuple, chunk2: tuple, overlap_distance: int=-1):
#     (_, s1, e1), (_, s2, e2) = chunk1, chunk2
#     if e1 <= s2:
#         return s2 - e1
#     elif e2 <= s1:
#         return s1 - e2
#     else:
#         return overlap_distance