import os
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# Lista de archivos con los que estamos trabajando
file_paths = [
    '1-gram.txt',
    '2-gram-nostop.txt',
    '2-gram-stop.txt',
    '3-gram-nostop.txt',
    '3-gram-stop.txt',
    '4-gram-nostop.txt',
    '4-gram-stop.txt'
]

# Función para cargar n-gramas desde archivos
def load_ngrams(filename):
    ngram_data = defaultdict(Counter)
    with open(filename, 'r') as f:
        current_model = None
        for line in f:
            line = line.strip()
            if line.startswith("Model:"):
                current_model = line.split(":")[1].strip()
            elif line and current_model:
                ngram, freq = line.rsplit(":", 1)
                ngram_data[current_model][ngram.strip()] = int(freq)
    return ngram_data

# Cargar todos los archivos generados
files = {filename: load_ngrams(filename) for filename in file_paths}

# Función para construir el espacio vectorial
def build_vector_space(ngram_data):
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
    model_texts = {}
    for model, ngram_counts in ngram_data.items():
        ngrams_text = ' '.join([f"{ngram} " * freq for ngram, freq in ngram_counts.items()])
        model_texts[model] = ngrams_text
    vector_space = vectorizer.fit_transform(model_texts.values())
    models = list(model_texts.keys())
    return vector_space, models, vectorizer

# Funciones de similitud
def cosine_sim(matrix):
    return cosine_similarity(matrix)

def jaccard_sim(matrix):
    matrix_binary = (matrix > 0).astype(int)
    n = matrix.shape[0]
    jaccard_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            jaccard_value = 1 - jaccard(matrix_binary[i].toarray().ravel(), matrix_binary[j].toarray().ravel())
            jaccard_matrix[i, j] = jaccard_matrix[j, i] = jaccard_value
    return jaccard_matrix

def angle_between_vectors(matrix):
    cos_sim = cosine_similarity(matrix)
    angles = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)
    return angles

# Asegurar que las gráficas y el archivo CSV se guarden en la carpeta del script
output_folder = os.path.dirname(os.path.abspath(__file__))

# Modificar la función de graficar para guardar las imágenes en escala de grises
def plot_similarity_matrix(matrix, models, title, filename):
    if matrix.size == 0 or matrix.ndim != 2:
        print(f"Error: Expected a non-empty 2D matrix for {title}, but got shape {matrix.shape}")
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=models, yticklabels=models, annot=True, fmt=".2f", cmap="Greys")
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel("Models")
    plt.tight_layout()
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

# Archivo CSV para guardar resultados
csv_filepath = os.path.join(output_folder, "similarity_results.csv")

# Escribir encabezado del CSV
with open(csv_filepath, mode='w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["File", "Metric", "Model 1", "Model 2", "Value"])

# Procesar cada archivo y calcular similitudes
for filename, ngram_data in files.items():
    vector_space, models, vectorizer = build_vector_space(ngram_data)
    vector_space = normalize(vector_space, norm='l2', axis=1)
    
    # Calcular y guardar cada una de las métricas de similitud
    cosine_sim_matrix = cosine_sim(vector_space)
    plot_similarity_matrix(cosine_sim_matrix, models, f"Cosine Similarity - {filename}", f"cosine_similarity_{filename}.png")
    
    jaccard_sim_matrix = jaccard_sim(vector_space)
    plot_similarity_matrix(jaccard_sim_matrix, models, f"Jaccard Similarity - {filename}", f"jaccard_similarity_{filename}.png")
    
    angle_matrix = angle_between_vectors(vector_space)
    plot_similarity_matrix(angle_matrix, models, f"Angle Between Vectors (Degrees) - {filename}", f"angle_similarity_{filename}.png")
    
    # Guardar resultados en el archivo CSV
    with open(csv_filepath, mode='a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Guardar similitudes coseno
        for i in range(len(models)):
            for j in range(len(models)):
                csv_writer.writerow([filename, "Cosine Similarity", models[i], models[j], cosine_sim_matrix[i, j]])
        
        # Guardar similitudes Jaccard
        for i in range(len(models)):
            for j in range(len(models)):
                csv_writer.writerow([filename, "Jaccard Similarity", models[i], models[j], jaccard_sim_matrix[i, j]])
        
        # Guardar ángulos
        for i in range(len(models)):
            for j in range(len(models)):
                csv_writer.writerow([filename, "Angle Between Vectors", models[i], models[j], angle_matrix[i, j]])

print(f"All results saved in {csv_filepath}")
