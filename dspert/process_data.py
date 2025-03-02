import json, os
from typing import List, Dict

def convert_bio_to_json(file_path: str, output_json_path: str = "output.json") -> List[Dict]:
    """
    Chuyển đổi file BIO định dạng CoNLL thành danh sách các mẫu và xuất ra file JSON.
    Tự động tạo thư mục đích nếu chưa tồn tại.

    Parameters
    ----------
    file_path : str
        Đường dẫn đến file BIO với token và nhãn phân cách bằng dấu cách, mỗi dòng là một cặp token-nhãn.
        Các mẫu được phân tách bằng dòng trống.
    output_json_path : str, optional
        Đường dẫn file JSON để lưu kết quả (mặc định là "output.json").

    Returns
    -------
    List[Dict]
        Danh sách các mẫu, mỗi mẫu có định dạng {'tokens': List[str], 'chunks': List[Tuple[str, int, int]]}.

    Raises
    ------
    FileNotFoundError
        Nếu file_path không tồn tại.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' not exists.")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    samples = []
    current_tokens = []
    current_tags = []
    
    for line in lines:
        line = line.strip()
        if line:  # Nếu dòng không trống
            token, tag = line.rsplit(' ', 1)  # Tách token và nhãn từ cuối dòng
            current_tokens.append(token)
            current_tags.append(tag)
        else:  # Dòng trống, kết thúc mẫu hiện tại
            if current_tokens:  # Chỉ thêm mẫu nếu có dữ liệu
                samples.append((current_tokens, current_tags))
                current_tokens = []
                current_tags = []
    
    # Thêm mẫu cuối nếu còn
    if current_tokens:
        samples.append((current_tokens, current_tags))
    
    # Chuyển đổi mỗi mẫu thành định dạng {'tokens': ..., 'chunks': ...}
    input_data = []
    for tokens, tags in samples:
        chunks = []
        current_label = None
        start_idx = None
        
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if current_label is not None:
                    chunks.append((current_label, start_idx, i))
                current_label = tag[2:]
                start_idx = i
            elif tag.startswith('I-'):
                if current_label == tag[2:] and start_idx is not None:
                    continue
                else:
                    if current_label is not None:
                        chunks.append((current_label, start_idx, i))
                    current_label = None
            else:  # O
                if current_label is not None:
                    chunks.append((current_label, start_idx, i))
                current_label = None
        
        if current_label is not None:
            chunks.append((current_label, start_idx, len(tokens)))
        
        input_data.append({
            'tokens': tokens,
            'chunks': chunks
        })
    
    output_dir = os.path.dirname(output_json_path)
    if output_dir:  # Chỉ tạo nếu output_json_path có thư mục cha
        os.makedirs(output_dir, exist_ok=True)
    
    # Xuất dữ liệu ra file JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)
    
    return input_data

if __name__=="__main__":
    convert_bio_to_json("data/origin_data/phoner_covid19/test.conll", "data/processed_data/phoner_covid19/test.json")
    convert_bio_to_json("data/origin_data/phoner_covid19/dev.conll", "data/processed_data/phoner_covid19/dev.json")
    convert_bio_to_json("data/origin_data/phoner_covid19/train.conll", "data/processed_data/phoner_covid19/train.json")