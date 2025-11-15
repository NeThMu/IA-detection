
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar vectores preprocesados
with open("bert_vectors_education.json", "r", encoding="utf-8") as f:
    bert_vectors = json.load(f)

with open("w2v_vectors_education.json", "r", encoding="utf-8") as f:
    w2v_vectors = json.load(f)

# Función para comparar un nuevo vector con el promedio humano vs IA
def compute_similarity_with_corpus(vector, corpus_vectors):
    corpus_matrix = np.array(corpus_vectors)
    vector = np.array(vector).reshape(1, -1)
    sim = cosine_similarity(vector, corpus_matrix)
    return float(np.mean(sim))


# Requiere: compute_similarity_with_corpus, bert_vectors, w2v_vectors definidos en el entorno

from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
import torch
from nltk.tokenize import word_tokenize

# Modelos
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
bert_model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# Asumimos que tienes un corpus base para entrenar temporalmente Word2Vec
# Para producción, carga un modelo entrenado previamente
def get_temp_w2v_model(corpus):
    tokenized = [word_tokenize(t.lower()) for t in corpus]
    model = Word2Vec(sentences=tokenized, vector_size=100, min_count=2)
    return model

def get_bert_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def get_w2v_vector(text, model):
    tokens = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Dado un texto, obtener vectores y calcular similitudes
def get_vector_scores_for_text(text, w2v_model):
    bert_vec = get_bert_vector(text)
    w2v_vec = get_w2v_vector(text, w2v_model)

    bert_sim = compute_similarity_with_corpus(bert_vec, bert_vectors)
    w2v_sim = compute_similarity_with_corpus(w2v_vec, w2v_vectors)

    return bert_sim, w2v_sim
