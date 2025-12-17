# app.py
import streamlit as st
from pyvis.network import Network
import tempfile
import os

from nltk.tokenize import sent_tokenize

from utils_word_graph import (
    extract_text_from_pdf,
    preprocess_sentence,
    build_word_graph,
    detect_communities
)

st.set_page_config(layout="wide")
st.title("📄 Word Graph, PageRank & Community Detection (Louvain)")

# =========================
# 1. Load PDF dari PATH
# =========================
PDF_PATH = "happiness.pdf"
st.info(f"PDF path: `{PDF_PATH}`")

if not os.path.exists(PDF_PATH):
    st.error("File PDF tidak ditemukan!")
    st.stop()

# =========================
# 2. Proses NLP
# =========================
with st.spinner("Memproses PDF & membangun word graph..."):
    raw_text = extract_text_from_pdf(PDF_PATH)
    sentences = sent_tokenize(raw_text)
    sentences_tokens = [
        preprocess_sentence(s) for s in sentences if len(s) > 1
    ]

    G, pagerank_scores = build_word_graph(
        sentences_tokens,
        weight_threshold=1
    )

st.success(
    f"Graph: {G.number_of_nodes()} nodes | {G.number_of_edges()} edges"
)

# =========================
# 3. FULL WORD GRAPH (PyVis)
# =========================
st.subheader("🕸️ Word Graph (FULL – Semua Token)")

net_full = Network(
    height="700px",
    width="100%",
    bgcolor="#ffffff",
    font_color="black"
)

for node in G.nodes():
    net_full.add_node(
        node,
        label=node,
        value=pagerank_scores[node] * 10000
    )

for u, v, d in G.edges(data=True):
    net_full.add_edge(u, v, value=d["weight"])

tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
net_full.save_graph(tmp_file.name)

st.components.v1.html(
    open(tmp_file.name, "r", encoding="utf-8").read(),
    height=750,
    scrolling=True
)

# =========================
# 4. Community Detection (Louvain)
# =========================
st.subheader("🧩 Community Detection (Louvain)")

partition = detect_communities(G)
communities = sorted(set(partition.values()))

selected_community = st.selectbox(
    "Pilih Community ID",
    communities
)

# ambil node dalam community terpilih
community_nodes = [
    n for n, c in partition.items()
    if c == selected_community
]

subG = G.subgraph(community_nodes).copy()

st.write(
    f"Community {selected_community}: "
    f"{subG.number_of_nodes()} nodes | "
    f"{subG.number_of_edges()} edges"
)

# =========================
# 5. Word Community Graph (PyVis)
# =========================
st.subheader("🎨 Word Community Graph (Interaktif)")

net_comm = Network(
    height="700px",
    width="100%",
    bgcolor="#ffffff",
    font_color="black"
)

for node in subG.nodes():
    net_comm.add_node(
        node,
        label=node,
        value=pagerank_scores[node] * 10000,
        color="#1f77b4"
    )

for u, v, d in subG.edges(data=True):
    net_comm.add_edge(u, v, value=d["weight"])

tmp_comm = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
net_comm.save_graph(tmp_comm.name)

st.components.v1.html(
    open(tmp_comm.name, "r", encoding="utf-8").read(),
    height=750,
    scrolling=True
)
