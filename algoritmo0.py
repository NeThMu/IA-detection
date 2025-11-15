import json
import re
from collections import Counter, defaultdict
from nltk import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

category_keywords = {
    "Technology": {
        "develop", "code", "program", "compute", "analyze", "encrypt", "secure", "automate", "innovate", "design",
        "engineer", "integrate", "optimize", "debug", "deploy", "test", "upgrade", "maintain", "network", "connect",
        "process", "store", "retrieve", "visualize", "simulate", "model", "calculate", "synchronize", "stream",
        "monitor", "control", "protect", "scan", "detect", "predict", "learn", "train", "classify", "recognize",
        "generate", "transform", "compress", "decompress", "render", "animate", "interact", "communicate", "transmit",
        "receive", "broadcast", "archive", "backup", "restore", "virtualize", "containerize", "orchestrate", "scale",
        "migrate", "synchronize", "authenticate", "authorize", "validate", "verify", "certify", "audit", "comply",
        "govern", "manage", "administer", "configure", "customize", "personalize", "localize", "globalize", "monetize",
        "license", "subscribe", "upgrade", "downgrade", "migrate", "export", "import", "sync", "share", "collaborate",
        "publish", "distribute", "license", "monetize", "analyze", "predict", "forecast", "optimize", "automate"
    },
    "Health": {
        "treat", "diagnose", "prevent", "cure", "heal", "recover", "rehabilitate", "exercise", "medicate", "vaccinate",
        "operate", "transplant", "reconstruct", "rehabilitate", "nurse", "care", "monitor", "measure", "analyze",
        "prescribe", "recommend", "advise", "consult", "examine", "scan", "image", "test", "screen", "detect",
        "prevent", "control", "manage", "improve", "enhance", "strengthen", "restore", "stabilize", "revive",
        "resuscitate", "sterilize", "disinfect", "sanitize", "immunize", "inoculate", "quarantine", "isolate",
        "research", "develop", "experiment", "observe", "document", "report", "educate", "inform", "train", "specialize"
    },
    "Education": {
        "teach", "learn", "study", "educate", "train", "instruct", "guide", "mentor", "coach", "explain",
        "demonstrate", "illustrate", "clarify", "simplify", "analyze", "evaluate", "assess", "grade", "score",
        "research", "experiment", "observe", "document", "report", "present", "discuss", "debate", "argue", "question",
        "answer", "solve", "calculate", "read", "write", "compose", "summarize", "paraphrase", "translate", "interpret",
        "create", "design", "develop", "innovate", "collaborate", "communicate", "network", "share", "publish",
        "distribute", "license", "certify", "accredit", "qualify", "specialize", "major", "minor", "graduate", "enroll"
    },
    "Business": {
        "manage", "lead", "organize", "plan", "strategize", "execute", "implement", "operate", "control", "monitor",
        "analyze", "evaluate", "optimize", "improve", "innovate", "develop", "design", "market", "sell", "advertise",
        "promote", "brand", "network", "negotiate", "contract", "deal", "trade", "export", "import", "invest", "finance",
        "budget", "account", "audit", "tax", "comply", "regulate", "govern", "risk", "insure", "protect", "secure",
        "certify", "accredit", "license", "partner", "collaborate", "communicate", "present", "report", "document",
        "train", "mentor", "coach", "recruit", "hire", "fire", "retain", "motivate", "reward", "compensate", "evaluate"
    },
    "Entertainment": {
        "entertain", "amuse", "delight", "enjoy", "laugh", "play", "perform", "act", "sing", "dance", "compose",
        "create", "produce", "direct", "edit", "film", "record", "broadcast", "stream", "publish", "distribute",
        "license", "market", "promote", "advertise", "brand", "network", "collaborate", "compete", "win", "lose",
        "celebrate", "award", "nominate", "critique", "review", "analyze", "evaluate", "recommend", "suggest", "watch",
        "listen", "read", "write", "illustrate", "animate", "design", "develop", "innovate", "experiment", "improvise",
        "rehearse", "practice", "train", "coach", "mentor", "inspire", "motivate", "engage", "interact", "communicate"
    }
}

context_indices = {
    "Technology": 1,
    "Health": 2,
    "Education": 3,
    "Business": 4,
    "Entertainment": 5
}

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text.lower())
    return tokens

def lemmatize_tokens(tokens):
    lemmatized_tokens = []
    for token, tag in nltk.pos_tag(tokens):
        pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, pos) if pos else lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)
    return lemmatized_tokens

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return None

def determine_category(tokens):
    category_scores = {category: sum(1 for word in tokens if word in keywords)
                       for category, keywords in category_keywords.items()}
    return max(category_scores, key=category_scores.get) if any(category_scores.values()) else None

term_doc_context_table = defaultdict(lambda: defaultdict(list))
context_counts = Counter()
terms_by_origin = {"human": defaultdict(Counter), "AI": defaultdict(Counter)}

with open('corpus.jsonl', 'r') as f:
    for doc_idx, line in enumerate(f):
        data = json.loads(line.strip())
        text = data['text']

        if 'domain' not in data or data['domain'] not in {"ingles", "outfox"}:
            continue

        model = data.get('model', 'unknown')
        is_human = model.lower() == "human"

        tokens = preprocess_text(text)
        lemmatized_tokens = lemmatize_tokens(tokens)
        category = determine_category(lemmatized_tokens)

        if category:
            context_counts[category] += 1

            for term in lemmatized_tokens:
                term_doc_context_table[term][doc_idx].append(context_indices[category])
                
                if is_human:
                    terms_by_origin["human"][category][term] += 1
                else:
                    terms_by_origin["AI"][category][term] += 1

most_common_context = context_counts.most_common(1)[0][0] if context_counts else None

with open('context_counts.txt', 'w') as f:
    for category, count in context_counts.items():
        f.write(f"{category}: {count}\n")


def store_human_special_terms(terms_by_origin, most_common_context, min_frequency=3):
    """Guarda términos exclusivos de humanos dentro del contexto más frecuente."""
    if not most_common_context:
        print("No hay contexto frecuente identificado.")
        return

    human_terms = terms_by_origin["human"].get(most_common_context, Counter())
    ai_terms = terms_by_origin["AI"].get(most_common_context, Counter())

    special_human_terms = [term for term, freq in human_terms.items() if freq < min_frequency and term not in ai_terms]

    with open("special_terms_human.txt", "w") as f:
        for term in special_human_terms:
            f.write(term + "\n")

    print(f"Se han almacenado {len(special_human_terms)} términos exclusivos de humanos en el contexto '{most_common_context}' en 'special_terms_human.txt'.")

store_human_special_terms(terms_by_origin, most_common_context)
