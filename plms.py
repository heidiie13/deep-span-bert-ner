from transformers import AutoModel, AutoTokenizer

model_name_list = ["roberta-base","bert-base-uncased","vinai/phobert-base", "xlm-roberta-base"]

for model_name in model_name_list:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"assets/transformers/{model_name}")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(f"assets/transformers/{model_name}")
    