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
from collections import Counter
from itertools import combinations
from community.community_louvain import best_partition

# ===============================
# KONFIGURASI
# ===============================
PDF_PATH = "happiness.pdf" 

# ===============================
# DOWNLOAD RESOURCE NLTK (CLOUD)
# ===============================
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("words")

download_nltk()

# ===============================
# STOPWORDS
# ===============================
stop_id = set(stopwords.words("indonesian"))
stop_en = set(stopwords.words("english"))
STOP_WORDS = stop_id.union(stop_en)

ENGLISH_WORDS = set(nltk_words.words("en"))

# ===============================
# FUNGSI (IDENTIK LOGIKA NOTEBOOK)
# ===============================
def extract_text_from_pdf(pdf_path: str) -> str:
    st.write(f"📄 Memproses PDF: `{pdf_path}`")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    st.write(f"📝 Panjang teks: {len(md_text)} karakter")
    return md_text


def preprocess_sentence(sentence: str):
    tokens = word_tokenize(sentence)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if t in ENGLISH_WORDS]
    return tokens


def build_cooccurrence_matrix(sentences_tokens):
    pair_counts = Counter()

    for sent in sentences_tokens:
        unique_words = list(set(sent))
        for w1, w2 in combinations(unique_words, 2):
            if w1 == w2:
                continue
            pair = tuple(sorted((w1, w2)))
            pair_counts[pair] += 1

    vocab = sorted({w for pair in pair_counts for w in pair})
    N = len(vocab)

    cooc_matrix = np.zeros((N, N), dtype=int)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    for (w1, w2), c in pair_counts.items():
        i, j = word_to_idx[w1], word_to_idx[w2]
        cooc_matrix[i, j] = c
        cooc_matrix[j, i] = c

    cooc_df = pd.DataFrame(cooc_matrix, index=vocab, columns=vocab)

    st.write(f"📌 Vocab size: {N}")
    st.write(f"📌 Jumlah pasangan kata: {len(pair_counts)}")

    return vocab, cooc_df, pair_counts


def build_word_graph_and_pagerank(vocab, pair_counts, weight_threshold=0):
    G = nx.Graph()
    G.add_nodes_from(vocab)

    for (w1, w2), c in pair_counts.items():
        if c > weight_threshold:
            G.add_edge(w1, w2, weight=c)

    pagerank_scores = nx.pagerank(G, weight="weight")

    st.write(
        f"🕸️ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G, pagerank_scores


def visualize_word_graph(G, pagerank_scores, top_n=30):
    selected_nodes = [
        w for w, _ in sorted(
            pagerank_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
    ]

    subG = G.subgraph(selected_nodes).copy()
    pos = nx.spring_layout(subG, k=0.6, iterations=100)

    pr_values = np.array([pagerank_scores[n] for n in subG.nodes()])
    node_sizes = 300 + 4000 * (pr_values - pr_values.min()) / (
        pr_values.max() - pr_values.min() + 1e-9
    )

    weights = [subG[u][v]["weight"] for u, v in subG.edges()]
    edge_widths = [1 + 5 * (w / max(weights)) for w in weights]

    plt.figure(figsize=(14, 14))
    nx.draw_networkx_nodes(subG, pos, node_size=node_sizes)
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.3)
    nx.draw_networkx_labels(subG, pos, font_size=9)
    plt.title(f"Word Graph Top {top_n} (PageRank)")
    plt.axis("off")

    st.pyplot(plt)
    plt.clf()


def visualize_communities(G, partition):
    pos = nx.spring_layout(G, k=0.6, iterations=100)
    num_communities = len(set(partition.values()))
    cmap = cm.get_cmap("viridis", num_communities)

    node_colors = [
        cmap(partition[n] / (num_communities - 1)) for n in G.nodes()
    ]

    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title("Community Detection (Louvain)")
    plt.axis("off")

    st.pyplot(plt)
    plt.clf()

# ===============================
# STREAMLIT APP
# ===============================
st.title("📄 Word Graph NLP (PDF dari Path)")
st.write("Analisis Co-occurrence, PageRank, dan Louvain")

top_n = st.slider("Top-N PageRank", 10, 100, 30)
weight_threshold = st.slider("Threshold Co-occurrence", 0, 5, 0)

if st.button("🚀 Jalankan Analisis"):
    with st.spinner("Memproses PDF..."):
        raw_text = extract_text_from_pdf(PDF_PATH)

        raw_sentences = sent_tokenize(raw_text)
        sentences_tokens = [
            preprocess_sentence(s) for s in raw_sentences
        ]
        sentences_tokens = [s for s in sentences_tokens if len(s) > 1]

        vocab, cooc_df, pair_counts = build_cooccurrence_matrix(sentences_tokens)
        G, pagerank_scores = build_word_graph_and_pagerank(
            vocab, pair_counts, weight_threshold
        )

    st.success("Analisis selesai ✅")

    st.subheader("📊 Top PageRank Words")
    pr_df = (
        pd.DataFrame(
            [{"word": w, "pagerank": s} for w, s in pagerank_scores.items()]
        )
        .sort_values("pagerank", ascending=False)
        .head(top_n)
    )
    st.dataframe(pr_df)

    st.subheader("🕸️ Word Graph")
    visualize_word_graph(G, pagerank_scores, top_n)

    st.subheader("🧩 Community Detection")
    partition = best_partition(G)
    visualize_communities(G, partition)
