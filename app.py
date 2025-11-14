import os
from pathlib import Path
from typing import List, Any, Dict

import streamlit as st
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
except Exception:
    ChatGroq = None
    HumanMessage = None


# Directories
DATA_DIR = Path("data")
TEXT_DIR = DATA_DIR / "text_files"
PDF_DIR = DATA_DIR / "pdf"
VECTOR_DIR = DATA_DIR / "vector_store"
COLLECTION_NAME = "rag_documents"


@st.cache_resource(show_spinner=False)
def get_embedding_manager(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def get_vector_store():
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"description": "RAG documents"}
    )
    return client, collection


def split_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def add_documents_to_store(collection, docs: List[Any], embedder: SentenceTransformer):
    if not docs:
        return 0
    chunks = split_documents(docs)
    texts = [d.page_content for d in chunks]
    metadatas = [dict(d.metadata) for d in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
    ids = [f"doc_{i}_{len(texts)}" for i in range(len(texts))]
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return len(texts)


def init_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or ChatGroq is None or HumanMessage is None:
        return None
    try:
        return ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=700)
    except Exception:
        return None


def retrieve(collection, embedder: SentenceTransformer, query: str, top_k: int = 5):
    q_emb = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    items = []
    if results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for i, (doc_id, content, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
            items.append(
                {
                    "id": doc_id,
                    "content": content,
                    "metadata": meta,
                    "similarity": 1 - dist if dist is not None else None,
                    "rank": i + 1,
                }
            )
    return items


def generate_answer(llm, query: str, context: str) -> str:
    if llm is None:
        # fallback preview
        preview = context.strip().split("\n")
        return "(No LLM configured) Context preview:\n" + "\n".join(preview[:6])
    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}

Answer:"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


def save_uploaded_files(files: List[Any]) -> List[str]:
    saved = []
    for f in files:
        filename = f.name
        if filename.lower().endswith(".pdf"):
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            out = PDF_DIR / filename
        else:
            TEXT_DIR.mkdir(parents=True, exist_ok=True)
            out = TEXT_DIR / filename
        with open(out, "wb") as fp:
            fp.write(f.getbuffer())
        saved.append(str(out))
    return saved


def load_new_documents(paths: List[str]) -> List[Any]:
    docs: List[Any] = []
    for p in paths:
        path = Path(p)
        try:
            if path.suffix.lower() == ".pdf":
                try:
                    docs.extend(PyPDFLoader(str(path)).load())
                except Exception:
                    docs.extend(PyMuPDFLoader(str(path)).load())
            else:
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
        except Exception as e:
            st.warning(f"Failed to load {path.name}: {e}")
    return docs


def main():
    load_dotenv()
    st.set_page_config(page_title="RAG Application", page_icon="🤖", layout="wide")

    st.sidebar.title("RAG Controls")
    uploaded = st.sidebar.file_uploader("Upload PDFs or TXTs", type=["pdf", "txt"], accept_multiple_files=True)
    reindex = st.sidebar.button("Add to Index")
    st.sidebar.markdown("---")
    top_k = st.sidebar.slider("Top K", 1, 10, 5)
    st.sidebar.caption("Set GROQ_API_KEY in .env to enable LLM answers.")

    st.title("RAG Chat")
    st.caption("Ask questions against your documents.")

    embedder = get_embedding_manager()
    client, collection = get_vector_store()
    llm = init_llm()

    if uploaded and reindex:
        saved_paths = save_uploaded_files(uploaded)
        new_docs = load_new_documents(saved_paths)
        added = add_documents_to_store(collection, new_docs, embedder)
        st.success(f"Added {added} chunks from {len(saved_paths)} files.")

    user_query = st.chat_input("Type your question...")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    if user_query:
        st.session_state.messages.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving..."):
                results = retrieve(collection, embedder, user_query, top_k=top_k)
                if not results:
                    answer = "No relevant context found."
                else:
                    context = "\n\n".join(r["content"] for r in results)
                    answer = generate_answer(llm, user_query, context)
                st.markdown(answer)
                # Sources
                if results:
                    with st.expander("Sources"):
                        for r in results:
                            src = r["metadata"].get("source_file") or r["metadata"].get("source") or "unknown"
                            page = r["metadata"].get("page")
                            score = r.get("similarity")
                            st.write(f"- {src} (page {page})  score={score:.3f}" if score is not None else f"- {src} (page {page})")
        st.session_state.messages.append(("assistant", answer))


if __name__ == "__main__":
    main()


