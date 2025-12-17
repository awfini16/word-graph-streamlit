# utils_word_graph.py
from collections import Counter
from itertools import combinations

import pymupdf4llm
import nltk
import networkx as nx
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, words as nltk_words
from community.community_louvain import best_partition

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("words")

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
