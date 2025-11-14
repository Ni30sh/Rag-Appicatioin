RAG (Retrieval-Augmented Generation) – Minimal CLI App

This repository contains a simple RAG pipeline you can run from the terminal. It:

- loads TXT and PDF files from data/
- splits into chunks
- generates embeddings with SentenceTransformer
- stores vectors in a persistent ChromaDB store
- retrieves relevant chunks for your question
- optionally uses Groq LLM to generate an answer (if GROQ_API_KEY is set)


Project structure

- data/text_files – put .txt files here
- data/pdf – put .pdf files here
- data/vector_store – auto-created for Chroma persistence
- main.py – CLI entry point for the RAG app
- app.py – Streamlit web dashboard for chat UI


Prerequisites

- Python 3.11+
- Install dependencies:

  pip install -r requirements.txt


Optional: LLM (Groq)

If you want generated answers, provide your Groq API key (otherwise you’ll see a context preview). The app reads `.env` automatically:

1) Create a `.env` file in the project root with:

  GROQ_API_KEY=your_key_here

2) Alternatively (Windows), you can set an environment variable:

  setx GROQ_API_KEY "your_key_here"

Restart your terminal after setting environment variables.


Run the CLI

1) Place documents:
   - TXT files in data/text_files
   - PDF files in data/pdf

2) Start the CLI:

  python main.py

3) Commands:
   - Ask any question directly
   - reindex – rebuild the vector index after adding files
   - exit – quit


Run the Web Dashboard (Streamlit)

1) Start the app:

  streamlit run app.py

2) In the browser:
   - Upload PDFs/TXTs from the sidebar (Add to Index)
   - Ask questions in the chat input
   - If GROQ_API_KEY is present, answers come from LLM; otherwise a context preview is shown


Notes

- On first run, the app builds the vector index; subsequent runs reuse it.
- If no GROQ_API_KEY is present, the app prints a concise context-based preview instead of an LLM answer.
- You can safely delete data/vector_store to rebuild from scratch.

