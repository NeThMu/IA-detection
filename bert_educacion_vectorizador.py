
import json
from transformers import AutoTokenizer, AutoModel
import torch

# Cargar modelo BERT en español
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# Función para obtener el vector [CLS]
def get_bert_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vector = outputs.last_hidden_state[:, 0, :]
    return cls_vector.squeeze().tolist()

# Cargar textos educativos
with open("education_texts.json", "r", encoding="utf-8") as f:
    education_texts = json.load(f)

# Generar vectores
bert_vectors = [get_bert_vector(text) for text in education_texts]

# Guardar vectores
with open("bert_vectors_education.json", "w", encoding="utf-8") as f:
    json.dump(bert_vectors, f)

print(f"Se generaron {len(bert_vectors)} vectores BERT en 'bert_vectors_education.json'")
