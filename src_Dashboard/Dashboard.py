import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import gradio as gr

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

books = pd.read_csv(os.path.join(BASE_DIR, "../data/books_with_emotions.csv"))
books["large_thumbnail"] = books["thumbnail"].fillna("") + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


with open(os.path.join(BASE_DIR, "../data/tagged_description.txt"), "r", encoding="utf-8") as f:
    lines = f.readlines()

documents = [Document(page_content=line.strip()) for line in lines if line.strip()]


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings,
    collection_name="books_collection"
)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        return []

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"] if pd.notna(row["description"]) else ""
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + ("..." if len(truncated_desc_split) > 30 else "")

        authors_split = row["authors"].split(";") if pd.notna(row["authors"]) else ["Unknown"]
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0].strip()} and {authors_split[1].strip()}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(a.strip() for a in authors_split[:-1])}, and {authors_split[-1].strip()}"
        else:
            authors_str = authors_split[0].strip()

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# ── Custom CSS ──────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --ink:       #0f0e0d;
    --paper:     #f5f0e8;
    --cream:     #ede7d9;
    --gold:      #b8973a;
    --gold-light:#d4af6a;
    --rust:      #8b3a2a;
    --muted:     #7a7060;
    --border:    #d4c9b0;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--paper) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--ink) !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 56px 24px 40px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
    position: relative;
}
.app-header::before {
    content: "❧";
    display: block;
    font-size: 1.4rem;
    color: var(--gold);
    margin-bottom: 12px;
    letter-spacing: 0.2em;
}
.app-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: clamp(2.4rem, 5vw, 3.8rem) !important;
    font-weight: 300 !important;
    letter-spacing: 0.04em !important;
    color: var(--ink) !important;
    line-height: 1.1 !important;
    margin: 0 0 10px !important;
}
.app-title em {
    font-style: italic;
    color: var(--gold);
}
.app-subtitle {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 300 !important;
    color: var(--muted) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}

/* ── Controls row ── */
.controls-row {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 28px 32px;
    margin-bottom: 36px;
    display: flex;
    gap: 20px;
    align-items: flex-end;
}

/* Textbox */
.gr-textbox label span,
label span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
textarea, input[type="text"] {
    background: var(--paper) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.1rem !important;
    color: var(--ink) !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--gold) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(184,151,58,0.12) !important;
}

/* Dropdowns */
select, .gr-dropdown select {
    background: var(--paper) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--ink) !important;
    padding: 10px 14px !important;
}

/* Button */
button.primary, .gr-button-primary, button[variant="primary"] {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 13px 28px !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
    white-space: nowrap !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: var(--gold) !important;
    transform: translateY(-1px) !important;
}

/* ── Gallery ── */
.section-label {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.4rem !important;
    font-weight: 400 !important;
    color: var(--ink) !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 20px !important;
    padding-bottom: 10px !important;
    border-bottom: 1px solid var(--border) !important;
}

.gallery-wrap .thumbnail-item,
.gradio-gallery .gallery-item {
    border-radius: 2px !important;
    overflow: hidden !important;
    transition: transform 0.25s, box-shadow 0.25s !important;
    border: 1px solid var(--border) !important;
}
.gallery-wrap .thumbnail-item:hover,
.gradio-gallery .gallery-item:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 32px rgba(15,14,13,0.18) !important;
}

/* Caption */


.caption-label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
    line-height: 1.5 !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 32px;
    margin-top: 48px;
    border-top: 1px solid var(--border);
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.08em;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

# ── Build UI ────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="Bibliotheca — Book Recommender") as dashboard:

    gr.HTML("""
    <div class="app-header">
        <h1 class="app-title">Biblio<em>theca</em></h1>
        <p class="app-subtitle">Semantic Book Discovery Engine</p>
    </div>
    """)

    with gr.Row(elem_classes="controls-row"):
        user_query = gr.Textbox(
            label="Describe the book you are looking for",
            placeholder="e.g. A melancholic story about love and loss in post-war Europe…",
            lines=2,
            scale=4,
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Category",
            value="All",
            scale=1,
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Emotional Tone",
            value="All",
            scale=1,
        )
        submit_button = gr.Button("Discover Books", variant="primary", scale=1)

    gr.HTML('<p class="section-label">Curated Recommendations</p>')

    output = gr.Gallery(
        label="",
        columns=8,
        rows=2,
        object_fit="cover",
        height=480,
        show_label=False,
    )

    gr.HTML("""
    <div class="app-footer">
        Bibliotheca &nbsp;·&nbsp; Semantic search powered by sentence-transformers &nbsp;·&nbsp; ❧
    </div>
    """)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()