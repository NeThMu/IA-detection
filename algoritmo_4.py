
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import PyPDF2
import docx
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import json

from integracion_vectores import get_vector_scores_for_text, get_temp_w2v_model

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

csv_path = "similarity_results.csv"
try:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("El archivo CSV está vacío.")
except (FileNotFoundError, ValueError):
    df = pd.DataFrame(columns=["Model 1", "Model 2", "Value", "Text"])

try:
    with open("special_terms_human.txt", "r", encoding="utf-8") as f:
        human_terms = set(f.read().splitlines())
except FileNotFoundError:
    human_terms = set()

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def read_file(filepath):
    try:
        if filepath.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        elif filepath.endswith(".pdf"):
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif filepath.endswith(".docx"):
            doc = docx.Document(filepath)
            text = " ".join([para.text for para in doc.paragraphs])
        else:
            text = ""
        if not text.strip():
            raise ValueError("El archivo no contiene texto válido.")
        return text
    except Exception as e:
        raise ValueError(f"Error al leer el archivo: {str(e)}")

def analyze_text_lsa_lda(text, corpus):
    try:
        if not corpus:
            return 0, 0
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text] + corpus)
        lsa = TruncatedSVD(n_components=min(2, tfidf_matrix.shape[1]))
        lsa_features = lsa.fit_transform(tfidf_matrix)
        lda_vectorizer = CountVectorizer()
        lda_matrix = lda_vectorizer.fit_transform([text] + corpus)
        lda = LatentDirichletAllocation(n_components=min(2, lda_matrix.shape[1]), random_state=42)
        lda_features = lda.fit_transform(lda_matrix)
        return np.mean(lsa_features[0]), np.mean(lda_features[0])
    except Exception:
        return 0, 0

def detect_ai(text):
    if not text.strip():
        return "Error: El archivo no contiene texto válido."

    tokens = preprocess_text(text)
    processed_text = " ".join(tokens)
    human_term_count = sum(1 for token in tokens if token in human_terms)
    freq_score = (human_term_count / len(tokens)) if tokens else 0

    similarities = df[(df['Model 1'] == 'human') & (df['Model 2'].isin(['chatGPT', 'bloomz', 'cohere']))]
    jaccard_score = similarities['Value'].max() if not similarities.empty else 0
    if jaccard_score > 1:
        jaccard_score = jaccard_score / 100

    try:
        with open("education_texts.json", "r", encoding="utf-8") as f:
            corpus = json.load(f)
    except Exception:
        corpus = []

    lsa_score, lda_score = analyze_text_lsa_lda(processed_text, corpus)
    w2v_model_temp = get_temp_w2v_model(corpus)
    bert_similarity, w2v_similarity = get_vector_scores_for_text(processed_text, w2v_model_temp)

    dot_score = jaccard_score
    angle_score = 1 - jaccard_score

    final_score = (
        0.10 * freq_score +
        0.10 * jaccard_score +
        0.15 * w2v_similarity +
        0.20 * bert_similarity +
        0.10 * dot_score +
        0.10 * angle_score +
        0.125 * lda_score +
        0.125 * lsa_score
    ) * 100

    final_score = estirar_score(final_score)
    final_score = min(max(final_score, 0), 100)

    if final_score >= 70:
        return f"Probabilidad de ser Humano: {round(final_score, 2)}%"
    elif final_score <= 40:
        return f"Probabilidad de ser IA: {round(100 - final_score, 2)}%"
    else:
        return f"Probabilidad ambigua (mixto): {round(final_score, 2)}%"

def upload_file():
    try:
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("PDF Files", "*.pdf"), ("Word Documents", "*.docx")])
        if filepath:
            text = read_file(filepath)
            if text:
                result = detect_ai(text)
                messagebox.showinfo("Resultado", result)
            else:
                messagebox.showerror("Error", "No se pudo leer el archivo.")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")


def estirar_score(score):
    if score >= 50:
        return 50 + ((score - 50) / 50) ** 0.5 * 50
    else:
        return 50 - ((50 - score) / 50) ** 0.5 * 50


root = tk.Tk()
root.title("Detector de IA")
root.geometry("300x150")
btn_upload = tk.Button(root, text="Subir Archivo", command=upload_file, padx=10, pady=5)
btn_upload.pack(expand=True)
root.mainloop()
