import streamlit as st
import nltk
import pymupdf4llm
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, words as nltk_words
from collections import Counter, defaultdict
from itertools import combinations
from community.community_louvain import best_partition

# ======================================================
# KONFIGURASI
# ======================================================
PDF_PATH = "happiness.pdf"

# ======================================================
# NLTK DOWNLOAD (CLOUD SAFE)
# ======================================================
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("words")

download_nltk()

# ======================================================
# STOPWORDS
# ======================================================
STOP_WORDS = set(stopwords.words("indonesian")).union(
    set(stopwords.words("english"))
)
ENGLISH_WORDS = set(nltk_words.words("en"))

# ======================================================
# FUNGSI NLP & GRAPH
# ======================================================
def extract_text_from_pdf(pdf_path):
    text = pymupdf4llm.to_markdown(pdf_path)
    return text


def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if t in ENGLISH_WORDS]
    return tokens


def build_cooccurrence(sentences_tokens):
    pair_counts = Counter()

    for sent in sentences_tokens:
        unique_words = list(set(sent))
        for w1, w2 in combinations(unique_words, 2):
            pair_counts[tuple(sorted((w1, w2)))] += 1

    vocab = sorted({w for pair in pair_counts for w in pair})
    return vocab, pair_counts


def build_graph_and_pagerank(vocab, pair_counts, threshold):
    G = nx.Graph()
    G.add_nodes_from(vocab)

    for (w1, w2), c in pair_counts.items():
        if c > threshold:
            G.add_edge(w1, w2, weight=c)

    pagerank = nx.pagerank(
        G, weight="weight", max_iter=100, tol=1e-6
    )
    return G, pagerank

# ======================================================
# 🔧 VISUALISASI KOMUNITAS (FIXED & SUBGRAPH)
# ======================================================
def visualize_communities_fixed(G, partition, min_degree=2, min_size=20):
    # Filter node degree kecil
    G = G.subgraph([n for n in G.nodes() if G.degree(n) >= min_degree])

    # Kelompok komunitas
    comm_groups = defaultdict(list)
    for node, cid in partition.items():
        if node in G:
            comm_groups[cid].append(node)

    # Ambil komunitas besar saja
    selected_nodes = []
    for cid, nodes in comm_groups.items():
        if len(nodes) >= min_size:
            selected_nodes.extend(nodes)

    subG = G.subgraph(selected_nodes)

    # Layout
    pos = nx.spring_layout(
        subG,
        k=2 / np.sqrt(subG.number_of_nodes()),
        iterations=300,
        seed=42
    )

    num_comms = len(set(partition[n] for n in subG.nodes()))
    cmap = cm.get_cmap("tab20", num_comms)

    node_colors = [
        cmap(partition[n] % num_comms) for n in subG.nodes()
    ]

    weights = [subG[u][v]["weight"] for u, v in subG.edges()]
    w_min, w_max = min(weights), max(weights)
    edge_widths = [
        0.5 + 4 * (w - w_min) / (w_max - w_min + 1e-9)
        for w in weights
    ]

    plt.figure(figsize=(18, 18))
    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors,
        node_size=500,
        alpha=0.9
    )
    nx.draw_networkx_edges(
        subG, pos,
        width=edge_widths,
        alpha=0.4
    )
    nx.draw_networkx_labels(subG, pos, font_size=7)

    plt.title("Community Detection (Louvain) – Subgraph Visualization")
    plt.axis("off")
    st.pyplot(plt)
    plt.clf()

# ======================================================
# STREAMLIT APP
# ======================================================
st.set_page_config(layout="wide")
st.title("📄 Word Graph NLP – Community Subgraph")

top_n = st.slider("Top-N PageRank", 10, 100, 30)
threshold = st.slider("Threshold Co-occurrence", 0, 5, 2)

if st.button("🚀 Jalankan Analisis"):
    with st.spinner("Memproses PDF..."):
        text = extract_text_from_pdf(PDF_PATH)
        sentences = sent_tokenize(text)

        tokens = [
            preprocess_sentence(s) for s in sentences
        ]
        tokens = [s for s in tokens if len(s) > 1]

        vocab, pair_counts = build_cooccurrence(tokens)
        G, pagerank = build_graph_and_pagerank(
            vocab, pair_counts, threshold
        )

        partition = best_partition(G)

    st.success("Analisis selesai ✅")

    # ===============================
    # OUTPUT PAGE RANK
    # ===============================
    st.subheader("📊 Top PageRank Words")
    pr_df = (
        pd.DataFrame(
            [{"word": w, "pagerank": s} for w, s in pagerank.items()]
        )
        .sort_values("pagerank", ascending=False)
        .head(top_n)
    )
    st.dataframe(pr_df)

    # ===============================
    # VISUALISASI KOMUNITAS
    # ===============================
    st.subheader("🧩 Community Graph (Filtered & Readable)")
    visualize_communities_fixed(G, partition)
