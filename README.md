# 📚 Bibliotheca — Semantic Book Discovery Engine

> A semantic book recommendation system powered by sentence-transformers, emotion analysis, and an elegant Gradio interface.

## 🌟 Overview

**Bibliotheca** is an intelligent book recommender that goes beyond keyword search. Instead of matching exact titles or genres, it understands the *meaning* behind your query — finding books that match the mood, themes, and narrative style you describe in plain language.

You can filter results by **genre category** and sort by **emotional tone** (happy, sad, suspenseful, angry, or surprising) to surface books that match exactly how you want to feel while reading.

---

## ✨ Features

- 🔍 **Semantic Search** — Describe the book you want in natural language; the engine finds thematically similar books using vector embeddings.
- 🎭 **Emotion-Aware Filtering** — Sort recommendations by emotional tone: joy, sadness, fear, anger, or surprise.
- 📂 **Category Filtering** — Narrow results by genre or subject category.
- 🖼️ **Visual Gallery UI** — Browse recommendations as a polished book-cover gallery with descriptions.
- ⚡ **Fast Retrieval** — Powered by ChromaDB for efficient in-memory vector similarity search.

---

## 🗂️ Project Structure

```
bibliotheca/
│
├── data/
│   ├── books_cleaned.csv            # Raw cleaned book dataset
│   ├── books_with_categories.csv    # Books enriched with genre categories
│   ├── books_with_emotions.csv      # Books enriched with emotion scores
│   └── tagged_description.txt       # ISBN-tagged descriptions for vector indexing
│
├── src_Analysis/
│   ├── A_Data_Exploration.ipynb     # Exploratory data analysis
│   ├── B_Vector_Search.ipynb        # Vector search experimentation
│   ├── C_Text_Classification.ipynb  # Category classification pipeline
│   └── D_Sentiment_Analysis.ipynb   # Emotion scoring pipeline
│
├── src_Dashboard/
│   └── Dashboard.py                 # Main Gradio application
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🧠 How It Works

The system is built on a multi-stage pipeline:

### 1. Data Preparation
Book metadata (titles, authors, descriptions, thumbnails) is cleaned and stored in CSV format. Each book's description is prefixed with its ISBN13 identifier and saved to `tagged_description.txt` for indexing.

### 2. Text Classification
Book descriptions are classified into simplified genre categories (Fiction, Non-Fiction, Science, History, etc.) using a zero-shot or fine-tuned text classification model.

### 3. Emotion Scoring
Each book description is scored across five emotional dimensions using a sentiment/emotion analysis model:
- **Joy** 😊
- **Sadness** 😢
- **Fear** 😨 
- **Anger** 😠
- **Surprise** 😲

Scores are stored in `books_with_emotions.csv`.

### 4. Vector Indexing
Book descriptions are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a **ChromaDB** vector store for fast semantic retrieval.



## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash

# 1. Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
cd src_Dashboard
python Dashboard.py
```

The Gradio interface will launch at `http://localhost:7860` by default.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `gradio` | Web UI framework |
| `langchain-chroma` | Vector store (ChromaDB) |
| `langchain-huggingface` | HuggingFace embeddings integration |
| `sentence-transformers` | Text embedding model (`all-MiniLM-L6-v2`) |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `python-dotenv` | Environment variable management |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Create a `.env` file in the project root if any API keys are needed (e.g., for HuggingFace Hub):

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

---


## 📊 Analysis Notebooks

The `src_Analysis/` folder contains the full data science pipeline:

- **A_Data_Exploration.ipynb** — Visualizations and statistics of the book dataset (rating distributions, top categories, description lengths, etc.)
- **B_Vector_Search.ipynb** — Experiments and evaluation of semantic search quality with different embedding models.
- **C_Text_Classification.ipynb** — Category labeling pipeline; maps raw genre strings to simplified categories.
- **D_Sentiment_Analysis.ipynb** — Emotion scoring pipeline using transformer-based models; generates the `books_with_emotions.csv` output.

---

## 📄 License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.

---
