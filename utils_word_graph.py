# utils_word_graph.py
import os
os.environ["NETWORKX_BACKEND"] = "networkx"  # FORCE pure-python backend

from collections import Counter
from itertools import combinations

import pymupdf4llm
import nltk
import networkx as nx

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words as nltk_words
from community.community_louvain import best_partition

# =====================================================
# ENSURE NLTK RESOURCES (NO LookupError)
# =====================================================
def ensure_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/words": "words",
    }

    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_resources()

# =====================================================
# GLOBAL NLP CONFIG
# =====================================================
STOP_WORDS = set(stopwords.words("indonesian")).union(
    set(stopwords.words("english"))
)

ENGLISH_WORDS = set(nltk_words.words())

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract markdown text from PDF using pymupdf4llm
    """
    return pymupdf4llm.to_markdown(pdf_path)

# =====================================================
# TEXT PREPROCESSING
# =====================================================
def preprocess_sentence(sentence: str) -> list[str]:
    """
    Tokenize, normalize, and filter sentence tokens
    """
    tokens = word_tokenize(sentence)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if t in ENGLISH_WORDS]
    return tokens

# =====================================================
# BUILD WORD GRAPH + PAGERANK (PURE PYTHON)
# =====================================================
def build_word_graph(
    sentences_tokens: list[list[str]],
    weight_threshold: int = 0
):
    """
    Build co-occurrence word graph and compute PageRank
    (SciPy-free, safe for Streamlit Cloud)
    """
    pair_counts = Counter()

    for sent in sentences_tokens:
        unique_words = set(sent)
        for w1, w2 in combinations(unique_words, 2):
            pair = tuple(sorted((w1, w2)))
            pair_counts[pair] += 1

    vocab = sorted({w for pair in pair_counts for w in pair})

    G = nx.Graph()
    G.add_nodes_from(vocab)

    for (w1, w2), count in pair_counts.items():
        if count > weight_threshold:
            G.add_edge(w1, w2, weight=count)

    # ---- PURE PYTHON PageRank (NO SCIPY) ----
    pagerank_scores = nx.pagerank(
        G,
        alpha=0.85,
        weight="weight",
        max_iter=200,
        tol=1.0e-6,
    )

    return G, pagerank_scores

# =====================================================
# COMMUNITY DETECTION (LOUVAIN)
# =====================================================
def detect_communities(G: nx.Graph) -> dict:
    """
    Detect word communities using Louvain method
    """
    return best_partition(G)
# utils_word_graph.py
from collections import Counter
from itertools import combinations

import pymupdf4llm
import nltk
import networkx as nx

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words as nltk_words
from community.community_louvain import best_partition

# =========================
# FIX NLTK RESOURCES
# =========================
def ensure_nltk_resources():
    resources = [
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "corpora/stopwords",
        "corpora/words"
    ]
    for r in resources:
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r.split("/")[-1])

ensure_nltk_resources()
# =========================

STOP_WORDS = set(stopwords.words("indonesian")).union(
    set(stopwords.words("english"))
)
ENGLISH_WORDS = set(nltk_words.words())

def extract_text_from_pdf(pdf_path: str) -> str:
    return pymupdf4llm.to_markdown(pdf_path)

def preprocess_sentence(sentence: str):
    tokens = word_tokenize(sentence)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if t in ENGLISH_WORDS]
    return tokens

def build_word_graph(sentences_tokens, weight_threshold=0):
    pair_counts = Counter()

    for sent in sentences_tokens:
        unique_words = list(set(sent))
        for w1, w2 in combinations(unique_words, 2):
            pair = tuple(sorted((w1, w2)))
            pair_counts[pair] += 1

    vocab = sorted({w for pair in pair_counts for w in pair})

    G = nx.Graph()
    G.add_nodes_from(vocab)

    for (w1, w2), c in pair_counts.items():
        if c > weight_threshold:
            G.add_edge(w1, w2, weight=c)

    pagerank_scores = nx.pagerank(G, weight="weight")
    return G, pagerank_scores

def detect_communities(G):
    partition = best_partition(G)
    return partition
