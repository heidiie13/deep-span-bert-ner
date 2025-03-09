import json, os
from typing import List, Dict

def convert_bio_to_json(file_path: str, output_json_path: str = "output.json") -> List[Dict]:
    """
    Converts a CoNLL-formatted BIO file to a list of samples and writes it to a JSON file.
    Automatically creates the destination directory if it doesn't exist.

    Args:
        file_path (str): Path to the BIO file with token and label separated by spaces,
            each line is a token-label pair. Samples are separated by empty lines.
        output_json_path (str, optional): Path to the JSON file to write the result to
            (default is "output.json").

    Returns:
        List[Dict]: A list of samples, each sample has the format
            {'tokens': List[str], 'chunks': List[Tuple[str, int, int]]}.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' not exists.")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    samples = []
    current_tokens = []
    current_tags = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line:
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                token, tag = parts
                current_tokens.append(token)
                current_tags.append(tag)
            else: 
                print(f"Warning: Invalid line format at line {i}: '{line}'. Skipping this line.")        
        else:
            if current_tokens:
                samples.append((current_tokens, current_tags))
                current_tokens = []
                current_tags = []

    if current_tokens:
        samples.append((current_tokens, current_tags))
    
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
            else:
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
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False, indent=4)
    
    return input_data

def convert_vimq_to_json(file_path: str, output_json_path: str = "output.json") -> List[Dict]:
    """
    Converts vimq formatted data from a JSON file to a JSON format similar to convert_bio_to_json output.
    Creates the destination directory if it doesn't exist. Preserves underscores in Vietnamese tokens.

    Args:
        file_path (str): Path to the JSON file containing vimq format data with
            'sentence' and 'seq_label' fields for each sample.
        output_json_path (str, optional): Path to the JSON file to write the result to
            (default is "output.json").

    Returns:
        List[Dict]: A list of samples, each sample has the format
            {'tokens': List[str], 'chunks': List[Tuple[str, int, int]]}.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file '{file_path}' does not exist.")

    with open(file_path, 'r', encoding='utf-8') as f:
        vimq_data = json.load(f)
    
    processed_data = []
    
    for item in vimq_data:
        sentence = item['sentence']
        tokens = []
        current_token = ""
        
        for char in sentence:
            if char == ' ':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
        
        chunks = []
        for label_info in item['seq_label']:
            start_idx, end_idx, label = label_info
            chunks.append((label, start_idx, end_idx + 1))
        
        sample = {
            'tokens': tokens,
            'chunks': chunks
        }
        processed_data.append(sample)
    
    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    return processed_data

if __name__=="__main__":
    convert_bio_to_json("data/origin_data/phoner_covid19/test.conll", "data/processed_data/phoner_covid19/test.json")
    convert_bio_to_json("data/origin_data/phoner_covid19/dev.conll", "data/processed_data/phoner_covid19/dev.json")
    convert_bio_to_json("data/origin_data/phoner_covid19/train.conll", "data/processed_data/phoner_covid19/train.json")
    
    convert_bio_to_json("data/origin_data/phoner_covid19_syllable/test.conll", "data/processed_data/phoner_covid19_syllable/test.json")
    convert_bio_to_json("data/origin_data/phoner_covid19_syllable/dev.conll", "data/processed_data/phoner_covid19_syllable/dev.json")
    convert_bio_to_json("data/origin_data/phoner_covid19_syllable/train.conll", "data/processed_data/phoner_covid19_syllable/train.json")
    
    convert_bio_to_json("data/origin_data/vimedner/test.txt", "data/processed_data/vimedner/test.json")
    convert_bio_to_json("data/origin_data/vimedner/dev.txt", "data/processed_data/vimedner/dev.json")
    convert_bio_to_json("data/origin_data/vimedner/train.txt", "data/processed_data/vimedner/train.json")
    
    convert_vimq_to_json("data/origin_data/vimq/train.json", "data/processed_data/vimq/train.json")
    convert_vimq_to_json("data/origin_data/vimq/dev.json", "data/processed_data/vimq/dev.json")
    convert_vimq_to_json("data/origin_data/vimq/test.json", "data/processed_data/vimq/test.json")