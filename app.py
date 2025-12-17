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

st.set_page_config(layout="wide")

# ======================================================
# NLTK DOWNLOAD
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
# NLP FUNCTIONS
# ======================================================
def extract_text_from_pdf(pdf_path):
    return pymupdf4llm.to_markdown(pdf_path)


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
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    N = len(vocab)

    cooc_matrix = np.zeros((N, N), dtype=int)
    for (w1, w2), c in pair_counts.items():
        i, j = word_to_idx[w1], word_to_idx[w2]
        cooc_matrix[i, j] = c
        cooc_matrix[j, i] = c

    cooc_df = pd.DataFrame(cooc_matrix, index=vocab, columns=vocab)
    return vocab, pair_counts, cooc_df


def build_graph_and_pagerank(vocab, pair_counts, threshold):
    G = nx.Graph()
    G.add_nodes_from(vocab)

    for (w1, w2), c in pair_counts.items():
        if c > threshold:
            G.add_edge(w1, w2, weight=c)

    # === Tetap pakai SciPy backend ===
    pagerank_scores = nx.pagerank(G, weight="weight")

    return G, pagerank_scores

# ======================================================
# VISUALIZATION FUNCTIONS
# ======================================================
def visualize_word_graph(G, pagerank, mode, top_n):
    if mode == "Parsial (Top-N PageRank)":
        selected_nodes = [
            w for w, _ in sorted(
                pagerank.items(), key=lambda x: x[1], reverse=True
            )[:top_n]
        ]
        G = G.subgraph(selected_nodes)

    pos = nx.spring_layout(
        G,
        k=2 / np.sqrt(G.number_of_nodes()),
        iterations=200,
        seed=42
    )

    pr_values = np.array([pagerank[n] for n in G.nodes()])
    node_sizes = 300 + 4000 * (
        (pr_values - pr_values.min()) /
        (pr_values.max() - pr_values.min() + 1e-9)
    )

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    w_min, w_max = min(weights), max(weights)
    edge_widths = [
        0.5 + 4 * (w - w_min) / (w_max - w_min + 1e-9)
        for w in weights
    ]

    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f"Word Graph PageRank – {mode}")
    plt.axis("off")
    st.pyplot(plt)
    plt.clf()


def visualize_communities_fixed(G, partition, min_degree=2, min_size=20):
    G = G.subgraph([n for n in G.nodes() if G.degree(n) >= min_degree])

    comm_groups = defaultdict(list)
    for node, cid in partition.items():
        if node in G:
            comm_groups[cid].append(node)

    selected_nodes = []
    for cid, nodes in comm_groups.items():
        if len(nodes) >= min_size:
            selected_nodes.extend(nodes)

    subG = G.subgraph(selected_nodes)

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
    plt.title("Community Detection (Louvain) – Subgraph")
    plt.axis("off")
    st.pyplot(plt)
    plt.clf()

# ======================================================
# STREAMLIT UI
# ======================================================
st.title("📄 Word Graph NLP – PageRank & Community Analysis")

threshold = st.slider("Threshold Co-occurrence", 0, 5, 2)
top_n = st.slider("Top-N PageRank (untuk parsial)", 10, 100, 30)
graph_mode = st.radio(
    "Visualisasi Word Graph PageRank",
    ["FULL Graph", "Parsial (Top-N PageRank)"]
)

if st.button("🚀 Jalankan Analisis"):
    with st.spinner("Memproses PDF..."):
        text = extract_text_from_pdf(PDF_PATH)
        sentences = sent_tokenize(text)
        tokens = [preprocess_sentence(s) for s in sentences]
        tokens = [s for s in tokens if len(s) > 1]

        vocab, pair_counts, cooc_df = build_cooccurrence(tokens)
        G, pagerank = build_graph_and_pagerank(
            vocab, pair_counts, threshold
        )
        partition = best_partition(G)

    st.success("Analisis selesai ✅")

    # ===============================
    # DATA UNDERSTANDING
    # ===============================
    st.subheader("📊 Data Understanding")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jumlah Kalimat", len(sentences))
    col2.metric("Vocab Size", len(vocab))
    col3.metric("Jumlah Edge", G.number_of_edges())
    col4.metric("Jumlah Komunitas", len(set(partition.values())))

    # ===============================
    # CO-OCCURRENCE PREVIEW
    # ===============================
    st.subheader("🔍 Preview Sub-Matriks Co-occurrence (10×10)")
    st.dataframe(cooc_df.iloc[:10, :10])

    # ===============================
    # PAGE RANK TABLE
    # ===============================
    st.subheader("🏆 Top PageRank Words")
    pr_df = (
        pd.DataFrame(
            [{"word": w, "pagerank": s} for w, s in pagerank.items()]
        )
        .sort_values("pagerank", ascending=False)
        .head(top_n)
    )
    st.dataframe(pr_df)

    # ===============================
    # WORD GRAPH
    # ===============================
    st.subheader("🕸️ Word Graph PageRank")
    visualize_word_graph(G, pagerank, graph_mode, top_n)

    # ===============================
    # COMMUNITY GRAPH
    # ===============================
    st.subheader("🧩 Community Graph (Louvain)")
    visualize_communities_fixed(G, partition)
