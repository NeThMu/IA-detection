import matplotlib.pyplot as plt
from collections import defaultdict, Counter

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
files = {
    '1-gram.txt': load_ngrams('1-gram.txt'),
    '2-gram-nostop.txt': load_ngrams('2-gram-nostop.txt'),
    '2-gram-stop.txt': load_ngrams('2-gram-stop.txt'),
    '3-gram-nostop.txt': load_ngrams('3-gram-nostop.txt'),
    '3-gram-stop.txt': load_ngrams('3-gram-stop.txt'),
    '4-gram-nostop.txt': load_ngrams('4-gram-nostop.txt'),
    '4-gram-stop.txt': load_ngrams('4-gram-stop.txt')
}

# Función para obtener n-gramas comunes entre modelos
def get_common_ngrams(ngram_data, exclude_model=None):
    models = list(ngram_data.keys())
    if exclude_model:
        models.remove(exclude_model)
    common_ngrams = set(ngram_data[models[0]].keys())
    for model in models[1:]:
        common_ngrams &= set(ngram_data[model].keys())
    return common_ngrams

# Función para graficar n-gramas con frecuencia en la etiqueta
def plot_ngrams(common_ngrams, ngram_data, title):
    # Calcular frecuencia total de cada n-grama entre modelos relevantes
    ngram_counts = Counter({ngram: sum(ngram_data[model][ngram] for model in ngram_data if ngram in ngram_data[model]) for ngram in common_ngrams})
    top_ngrams = ngram_counts.most_common(10)  # Seleccionar el top 10 para la visualización

    # Preparar etiquetas con frecuencia
    ngram_labels = [f"{ngram} ({freq})" for ngram, freq in top_ngrams]
    frequencies = [freq for _, freq in top_ngrams]

    plt.figure(figsize=(10, 6))
    plt.barh(ngram_labels, frequencies)
    plt.xlabel("Frequency")
    plt.ylabel("N-gram")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# Procesar cada archivo y generar gráficos
for filename, ngram_data in files.items():
    # Obtener n-gramas comunes entre modelos no humanos
    non_human_common_ngrams = get_common_ngrams(ngram_data, exclude_model='human')
    # Graficar n-gramas comunes entre modelos no humanos
    plot_ngrams(non_human_common_ngrams, ngram_data, f"Common N-grams (Non-Human Models) - {filename}")

    # Obtener n-gramas comunes entre modelo humano y otros modelos
    human_common_ngrams = set(ngram_data['human'].keys())
    for model in ngram_data:
        if model != 'human':
            human_common_ngrams &= set(ngram_data[model].keys())
    # Graficar n-gramas comunes entre modelo humano y otros modelos
    plot_ngrams(human_common_ngrams, ngram_data, f"Common N-grams (Human & Other Models) - {filename}")
