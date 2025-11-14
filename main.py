import os
from pathlib import Path
from typing import List, Any, Dict

from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
except Exception:
    ChatGroq = None  # Optional
    HumanMessage = None  # Optional

DATA_DIR = Path("data")
TEXT_DIR = DATA_DIR / "text_files"
PDF_DIR = DATA_DIR / "pdf"
VECTOR_DIR = DATA_DIR / "vector_store"
COLLECTION_NAME = "rag_documents"


def find_documents() -> List[Any]:
    """Load TXT and PDF documents from data directories."""
    # Ensure .env is loaded so env vars (e.g., GROQ_API_KEY) are available
    load_dotenv()
    docs: List[Any] = []

    if TEXT_DIR.exists():
        text_loader = DirectoryLoader(
            str(TEXT_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        docs.extend(text_loader.load())

    if PDF_DIR.exists():
        # Load PDFs one-by-one with graceful fallback to PyMuPDFLoader if pypdf is missing
        for pdf_path in PDF_DIR.rglob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs.extend(loader.load())
            except ImportError:
                # Fallback to PyMuPDFLoader if pypdf is not installed
                try:
                    loader = PyMuPDFLoader(str(pdf_path))
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Error loading file {pdf_path}: {e}")
            except Exception as e:
                # If PyPDFLoader fails for other reasons, try PyMuPDFLoader as well
                try:
                    loader = PyMuPDFLoader(str(pdf_path))
                    docs.extend(loader.load())
                except Exception:
                    print(f"Error loading file {pdf_path}: {e}")

    return docs


def split_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


class EmbeddingManager:
    """Wrapper around SentenceTransformer for embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=True)


class VectorStore:
    """ChromaDB vector store wrapper."""

    def __init__(self, collection_name: str, persist_directory: Path):
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"description": "RAG documents"}
        )

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict], embeddings):
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_embedding, top_k: int = 5):
        return self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

    def count(self) -> int:
        return self.collection.count()


def build_or_update_index(embedding_manager: EmbeddingManager, vector_store: VectorStore):
    print("Loading documents from data/ ...")
    raw_docs = find_documents()
    if not raw_docs:
        print("No documents found in data/. Add PDFs to data/pdf or TXT to data/text_files.")
        return

    print(f"Loaded {len(raw_docs)} documents. Splitting into chunks ...")
    chunks = split_documents(raw_docs)
    print(f"Created {len(chunks)} chunks. Generating embeddings ...")

    texts = [d.page_content for d in chunks]
    metadatas = [dict(d.metadata) for d in chunks]
    embeddings = embedding_manager.encode(texts)

    ids = [f"doc_{i}" for i in range(len(chunks))]
    vector_store.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())
    print(f"Indexed {len(chunks)} chunks. Total in store: {vector_store.count()}")


def init_optional_llm():
    """Initialize Groq LLM if GROQ_API_KEY is set and package available."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or ChatGroq is None or HumanMessage is None:
        return None
    try:
        return ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0.1, max_tokens=600)
    except Exception:
        return None


def retrieve(vector_store: VectorStore, embedding_manager: EmbeddingManager, query: str, top_k: int = 5):
    q_emb = embedding_manager.encode([query])[0].tolist()
    results = vector_store.query(q_emb, top_k=top_k)
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
        return f"(No LLM configured) Context-based answer preview:\n{context[:800]}"
    prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {query}

Answer:"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    return getattr(resp, "content", str(resp))


def interactive_cli():
    print("Initializing RAG components ...")
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(COLLECTION_NAME, VECTOR_DIR)

    if vector_store.count() == 0:
        build_or_update_index(embedding_manager, vector_store)
    else:
        print(f"Vector store has {vector_store.count()} chunks. Skipping re-index.")

    llm = init_optional_llm()
    if llm is None:
        print("Tip: Set GROQ_API_KEY to enable LLM answers. Showing context previews instead.")

    print("\nRAG CLI ready. Type your question (or 'reindex', 'exit').")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if query.lower() == "reindex":
            build_or_update_index(embedding_manager, vector_store)
            continue

        if not query:
            continue

        results = retrieve(vector_store, embedding_manager, query, top_k=5)
        if not results:
            print("No results found.")
            continue

        context = "\n\n".join(r["content"] for r in results)
        answer = generate_answer(llm, query, context)
        print("\nAnswer:\n" + answer)
        
        print("\nTop sources:")
        for r in results[:3]:
            src = r["metadata"].get("source_file") or r["metadata"].get("source") or "unknown"
            page = r["metadata"].get("page")
            score = r.get("similarity")
            print(f"- {src} (page {page})  score={score:.3f}" if score is not None else f"- {src} (page {page})")
        print("\n" + "-" * 60)


def main():
    interactive_cli()


if __name__ == "__main__":
    main()
