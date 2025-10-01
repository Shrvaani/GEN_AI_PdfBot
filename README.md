# RAG PDF Chatbot - Deployment Guide

A Streamlit-based RAG chatbot that allows users to upload PDFs and ask questions using the Hugging Face `openai/gpt-oss-20b` model.

## Features

- ✅ Upload multiple PDFs (500+ pages supported)
- ✅ Multi-chat conversations with persistent history
- ✅ GPT-style chat interface
- ✅ Rename and delete chats
- ✅ Adjustable reasoning levels (Low/Medium/High)
- ✅ No local model downloads (uses HF Inference API)
- ✅ Theme-adaptive UI (works in light/dark modes)
- ✅ Similarity-based retrieval to reduce hallucinations

## Deployment on Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: RAG PDF Chatbot"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your GitHub repository
4. Set:
   - **Main file path**: `deploy/app.py`
   - **Python version**: 3.11
5. Click "Advanced settings" → Add secrets:
   ```
   HF_TOKEN = "hf_your_token_here"
   ```
6. Click "Deploy"

### 3. Get Your HF Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (Read access is sufficient)
3. Copy and paste it in Streamlit Cloud secrets

## Local Development

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_your_token_here
streamlit run app.py
```

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: openai/gpt-oss-20b (Hugging Face Inference API)
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (via HF API)
- **PDF Processing**: PyMuPDF
- **Vector Search**: In-memory NumPy cosine similarity
- **Persistence**: Local JSON file (~/.rag_pdfbot/chats.json)

## Model Reference

- [openai/gpt-oss-20b on Hugging Face](https://huggingface.co/openai/gpt-oss-20b)

