"""Streamlit PDF RAG application using LangChain, OpenAI, FAISS, and LangSmith."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_COUNT = 4
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LANGSMITH_PROJECT = "pdf-rag-streamlit-app"
DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
NOT_FOUND_RESPONSE = (
    "I could not find enough information in the uploaded PDF to answer this question."
)


def configure_environment() -> dict[str, Any]:
    """Load runtime configuration from .env and configure optional tracing."""
    load_dotenv(override=True, encoding="utf-8-sig")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    requested_tracing = (
        os.getenv("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "false"))
        .strip()
        .lower()
        == "true"
    )
    langsmith_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    langsmith_project = (
        os.getenv("LANGCHAIN_PROJECT")
        or os.getenv("LANGSMITH_PROJECT")
        or DEFAULT_LANGSMITH_PROJECT
    )
    langsmith_endpoint = os.getenv("LANGCHAIN_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT)
    tracing_enabled = requested_tracing and bool(langsmith_api_key)

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if tracing_enabled else "false"
    os.environ["LANGCHAIN_PROJECT"] = langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint

    return {
        "openai_api_key_present": bool(openai_api_key),
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "langsmith_requested": requested_tracing,
        "langsmith_enabled": tracing_enabled,
        "langsmith_project": langsmith_project,
        "langsmith_endpoint": langsmith_endpoint,
    }


def initialize_session_state() -> None:
    """Create Streamlit session-state keys used by the application."""
    defaults = {
        "pdf_fingerprint": None,
        "pdf_name": None,
        "vector_store": None,
        "retriever": None,
        "rag_chain": None,
        "chat_messages": [],
        "sources": [],
        "processing_status": "waiting",
        "page_count": 0,
        "chunk_count": 0,
        "uploader_key": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def calculate_file_hash(file_bytes: bytes) -> str:
    """Return a SHA-256 fingerprint for the uploaded PDF bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def load_pdf_documents(file_bytes: bytes, file_name: str) -> list[Document]:
    """Load PDF pages from uploaded bytes while preserving page metadata."""
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("Unsupported file type. Please upload a PDF document.")

    with tempfile.TemporaryDirectory(prefix="pdf_rag_") as temporary_directory:
        temporary_path = Path(temporary_directory) / "uploaded.pdf"
        temporary_path.write_bytes(file_bytes)

        loader = PyPDFLoader(str(temporary_path))
        documents = loader.load()

    for document in documents:
        document.metadata["source"] = file_name
        document.metadata["file_name"] = file_name

    meaningful_documents = [
        document for document in documents if document.page_content and document.page_content.strip()
    ]
    if not meaningful_documents:
        raise ValueError(
            "No extractable text was found. This PDF may contain scanned images and may require OCR."
        )

    return meaningful_documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split PDF pages into overlapping chunks while retaining metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]


def create_vector_store(chunks: list[Document], embeddings: OpenAIEmbeddings) -> FAISS:
    """Create an in-memory FAISS vector store from document chunks."""
    if not chunks:
        raise ValueError("The PDF did not produce any searchable text chunks.")

    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


def format_page_number(metadata: dict[str, Any]) -> str:
    """Format LangChain PDF page metadata for user display."""
    page = metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)
    if page is None:
        return "Unknown"
    return str(page)


def format_documents(documents: list[Document]) -> str:
    """Format retrieved documents as source-labeled prompt context."""
    formatted_chunks: list[str] = []
    for document in documents:
        file_name = document.metadata.get("file_name", document.metadata.get("source", "Uploaded PDF"))
        page_number = format_page_number(document.metadata)
        content = " ".join(document.page_content.split())
        if content:
            formatted_chunks.append(
                f"[Source: {file_name}, Page: {page_number}]\n{content}"
            )
    return "\n\n".join(formatted_chunks)


def create_rag_chain(retriever: Any, llm: ChatOpenAI) -> Any:
    """Create a modern Runnable-based RAG chain that returns answers and sources."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a PDF question-answering assistant.\n\n"
                    "Answer the user's question only using the supplied PDF context.\n"
                    "Do not use outside knowledge or invent information.\n"
                    f'If the context does not contain enough information to answer the question, respond: "{NOT_FOUND_RESPONSE}"\n'
                    "Keep the answer clear, factual, and concise."
                ),
            ),
            (
                "human",
                "PDF context:\n{context}\n\nUser question:\n{question}",
            ),
        ]
    ).with_config({"run_name": "pdf-grounded-prompt", "tags": ["streamlit-rag", "pdf-rag"]})

    answer_chain = (
        prompt
        | llm.with_config({"run_name": "openai-answer-generation", "tags": ["streamlit-rag"]})
        | StrOutputParser()
    ).with_config({"run_name": "pdf-question-answering", "tags": ["streamlit-rag", "pdf-rag"]})

    def retrieve(question: str) -> dict[str, Any]:
        source_documents = retriever.invoke(
            question,
            config={"run_name": "pdf-retrieval", "tags": ["streamlit-rag", "pdf-retrieval"]},
        )
        return {"question": question, "source_documents": source_documents}

    def answer(payload: dict[str, Any]) -> dict[str, Any]:
        source_documents = payload["source_documents"]
        context = format_documents(source_documents)
        if not context.strip():
            return {"answer": NOT_FOUND_RESPONSE, "source_documents": source_documents}

        response = answer_chain.invoke(
            {"question": payload["question"], "context": context},
            config={"run_name": "grounded-answer", "tags": ["streamlit-rag", "pdf-question-answering"]},
        )
        return {"answer": response, "source_documents": source_documents}

    return (
        RunnableLambda(retrieve).with_config(
            {"run_name": "retrieve-pdf-context", "tags": ["streamlit-rag", "pdf-retrieval"]}
        )
        | RunnableLambda(answer).with_config(
            {"run_name": "answer-with-pdf-context", "tags": ["streamlit-rag", "pdf-question-answering"]}
        )
    ).with_config({"run_name": "streamlit-rag", "tags": ["streamlit-rag", "pdf-rag"]})


def display_sources(source_documents: list[Document]) -> None:
    """Display unique source excerpts for a generated answer."""
    if not source_documents:
        st.info("No supporting sources were returned.")
        return

    seen: set[tuple[str, str, str]] = set()
    unique_sources: list[Document] = []
    for document in source_documents:
        file_name = str(document.metadata.get("file_name", document.metadata.get("source", "Uploaded PDF")))
        page_number = format_page_number(document.metadata)
        excerpt = " ".join(document.page_content.split())[:450]
        source_key = (file_name, page_number, excerpt)
        if excerpt and source_key not in seen:
            seen.add(source_key)
            unique_sources.append(document)

    with st.expander("Sources used", expanded=False):
        for index, document in enumerate(unique_sources, start=1):
            file_name = document.metadata.get("file_name", document.metadata.get("source", "Uploaded PDF"))
            page_number = format_page_number(document.metadata)
            excerpt = " ".join(document.page_content.split())[:450]
            st.markdown(f"**Source {index}**")
            st.caption(f"File: {file_name} | Page: {page_number}")
            st.write(excerpt)


def safe_error_message(error: Exception) -> str:
    """Return a user-safe error message without exposing credentials or stack traces."""
    message = str(error).strip()
    lowered = message.lower()

    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return "Authentication failed. Please check the API keys in your .env file."
    if "rate limit" in lowered or "quota" in lowered:
        return "The OpenAI request was rate-limited or quota-limited. Please try again later."
    if "password" in lowered or "encrypted" in lowered:
        return "This PDF appears to be password-protected or encrypted. Please upload an unlocked PDF."
    if "no extractable text" in lowered:
        return message
    if "unsupported file type" in lowered:
        return message
    if "searchable text chunks" in lowered:
        return message

    return "Something went wrong while processing the PDF or answering the question. Please try again."


def clear_document_state(clear_uploader: bool = False) -> None:
    """Clear document-specific state and optionally reset the file uploader widget."""
    st.session_state.pdf_fingerprint = None
    st.session_state.pdf_name = None
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.rag_chain = None
    st.session_state.chat_messages = []
    st.session_state.sources = []
    st.session_state.processing_status = "waiting"
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0
    if clear_uploader:
        st.session_state.uploader_key += 1


def reset_application() -> None:
    """Reset the uploaded document and current conversation."""
    clear_document_state(clear_uploader=True)


def process_uploaded_pdf(uploaded_file: Any, config: dict[str, Any]) -> None:
    """Process a newly uploaded PDF into an in-memory FAISS retriever and RAG chain."""
    file_bytes = uploaded_file.getvalue()
    file_fingerprint = calculate_file_hash(file_bytes)

    if (
        st.session_state.pdf_fingerprint == file_fingerprint
        and st.session_state.retriever is not None
        and st.session_state.rag_chain is not None
    ):
        return

    clear_document_state()
    st.session_state.pdf_fingerprint = file_fingerprint
    st.session_state.pdf_name = uploaded_file.name
    st.session_state.processing_status = "processing"

    if not config["openai_api_key_present"]:
        st.session_state.processing_status = "error"
        st.error("OPENAI_API_KEY is missing. Add it to your .env file before processing PDFs.")
        return

    try:
        with st.status("Processing PDF", expanded=True) as status:
            st.write("Extracting text and page metadata")
            documents = load_pdf_documents(file_bytes, uploaded_file.name)

            st.write("Splitting text into overlapping chunks")
            chunks = split_documents(documents)

            st.write("Creating OpenAI embeddings")
            embeddings = OpenAIEmbeddings(model=config["embedding_model"])

            st.write("Building FAISS vector store")
            vector_store = create_vector_store(chunks, embeddings)
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_COUNT},
            )

            st.write("Creating grounded RAG chain")
            llm = ChatOpenAI(model=config["chat_model"], temperature=0)
            rag_chain = create_rag_chain(retriever, llm)

            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.session_state.page_count = len(documents)
            st.session_state.chunk_count = len(chunks)
            st.session_state.processing_status = "ready"

            status.update(
                label="PDF processed",
                state="complete",
                expanded=False,
            )
            st.success(
                f"Processed {uploaded_file.name}: {len(documents)} page(s), {len(chunks)} chunk(s)."
            )
    except Exception as error:
        clear_document_state()
        st.session_state.processing_status = "error"
        st.error(safe_error_message(error))


def display_sidebar(config: dict[str, Any]) -> None:
    """Render sidebar workflow, configuration, and observability details."""
    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "1. Upload PDF\n"
            "2. Extract text\n"
            "3. Split into chunks\n"
            "4. Generate embeddings\n"
            "5. Store in FAISS\n"
            "6. Retrieve relevant content\n"
            "7. Generate a grounded answer"
        )

        st.header("Configuration")
        st.write(f"Chat model: `{config['chat_model']}`")
        st.write(f"Embedding model: `{config['embedding_model']}`")
        st.write(f"Chunk size: `{CHUNK_SIZE}`")
        st.write(f"Chunk overlap: `{CHUNK_OVERLAP}`")
        st.write(f"Retrieval count: `{RETRIEVAL_COUNT}`")

        st.header("Observability")
        if config["langsmith_enabled"]:
            st.success("LangSmith tracing: Enabled")
            st.write(f"Project: `{config['langsmith_project']}`")
        else:
            st.info("LangSmith tracing: Disabled")
            if config["langsmith_requested"]:
                st.warning(
                    "LangSmith tracing is not configured. The application will continue without traceability."
                )


def display_chat_history() -> None:
    """Render prior chat messages and their sources."""
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                display_sources(message.get("sources", []))


def answer_question(question: str) -> None:
    """Run retrieval and answer generation for a user question."""
    if not question.strip():
        st.warning("Please enter a question.")
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF and generating an answer"):
            try:
                result = st.session_state.rag_chain.invoke(
                    question,
                    config={"run_name": "streamlit-rag-question", "tags": ["streamlit-rag", "pdf-rag"]},
                )
                answer = result.get("answer", NOT_FOUND_RESPONSE)
                source_documents = result.get("source_documents", [])
                st.markdown(answer)
                display_sources(source_documents)

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_documents,
                    }
                )
                st.session_state.sources = source_documents
            except Exception as error:
                safe_message = safe_error_message(error)
                st.error(safe_message)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": safe_message, "sources": []}
                )


def main() -> None:
    """Run the Streamlit PDF RAG application."""
    st.set_page_config(
        page_title="PDF RAG Question Answering",
        page_icon=":page_facing_up:",
        layout="centered",
    )

    config = configure_environment()
    initialize_session_state()
    display_sidebar(config)

    st.title("PDF RAG Question-Answering App")
    st.write(
        "Upload one text-based PDF, then ask questions grounded only in the document content."
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.session_state.pdf_name:
            st.info(f"Current document: {st.session_state.pdf_name}")
    with col2:
        if st.button("Clear document", use_container_width=True):
            reset_application()
            st.rerun()

    if not config["openai_api_key_present"]:
        st.error("OPENAI_API_KEY is missing. Add it to your .env file to use the app.")

    if uploaded_file is None:
        st.info("Upload a PDF to begin.")
    else:
        process_uploaded_pdf(uploaded_file, config)

    if st.session_state.processing_status == "ready":
        st.success(
            f"Ready: {st.session_state.page_count} page(s), {st.session_state.chunk_count} chunk(s)."
        )
        display_chat_history()
        question = st.chat_input("Ask a question about the uploaded PDF")
        if question:
            answer_question(question)
    elif uploaded_file is not None and st.session_state.processing_status == "processing":
        st.info("The PDF is still being processed.")


if __name__ == "__main__":
    main()
