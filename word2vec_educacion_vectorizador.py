
import json
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np

nltk.download("punkt")

# Cargar textos educativos
with open("education_texts.json", "r", encoding="utf-8") as f:
    education_texts = json.load(f)

# Tokenizar el corpus completo
tokenized_corpus = [word_tokenize(text.lower()) for text in education_texts]

# Entrenar modelo Word2Vec
w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=2, workers=4)

# Obtener vector promedio por documento
def get_doc_vector(text, model):
    words = word_tokenize(text.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0).tolist() if vectors else [0.0] * model.vector_size

# Vectorizar todos los textos
w2v_vectors = [get_doc_vector(text, w2v_model) for text in education_texts]

# Guardar los vectores
with open("w2v_vectors_education.json", "w", encoding="utf-8") as f:
    json.dump(w2v_vectors, f)

print(f"Se generaron {len(w2v_vectors)} vectores Word2Vec en 'w2v_vectors_education.json'")
