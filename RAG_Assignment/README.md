# PDF RAG Question-Answering App

## Description

This Streamlit application lets you upload one PDF and ask grounded questions about its contents. It uses LangChain for the RAG pipeline, OpenAI for embeddings and answer generation, FAISS for local vector search, and optional LangSmith tracing for observability.

## Features

- PDF upload
- Automatic text extraction
- Text chunking
- OpenAI embeddings
- FAISS vector search
- Grounded PDF question answering
- Source-page display
- Chat history
- Document-change detection
- LangSmith tracing
- Secure environment-variable management
- Friendly error handling

## Architecture

```text
PDF Upload
   |
   v
PyPDFLoader
   |
   v
RecursiveCharacterTextSplitter
   |
   v
OpenAI Embeddings
   |
   v
FAISS Vector Store
   |
   v
LangChain Retriever
   |
   v
Prompt + Retrieved Context
   |
   v
OpenAI Chat Model
   |
   v
Grounded Answer + Sources
   |
   v
LangSmith Trace
```

## Project structure

```text
pdf-rag-app/
+-- app.py
+-- requirements.txt
+-- .env
+-- .gitignore
+-- README.md
```

The `.env` file is local-only and ignored by Git. It is shown here because the app reads it at runtime, but it should not be committed.

## Prerequisites

- Python 3.11 or later
- OpenAI API account and API key
- LangSmith account and API key for optional tracing
- Internet connection for OpenAI requests

## Windows PowerShell setup

```powershell
git clone <repository-url>
cd pdf-rag-app

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project uses the existing `.env` file in this folder. Confirm it contains your OpenAI key and any optional LangSmith values, then run:

```powershell
streamlit run app.py
```

## Windows Command Prompt setup

```bat
python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## macOS/Linux setup

```bash
git clone <repository-url>
cd pdf-rag-app

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Environment configuration

`OPENAI_API_KEY` is mandatory and must contain your OpenAI API key.

`OPENAI_CHAT_MODEL` controls the OpenAI chat model. The default is `gpt-4.1-mini`.

`OPENAI_EMBEDDING_MODEL` controls the OpenAI embedding model. The default is `text-embedding-3-small`.

`LANGCHAIN_TRACING_V2` enables LangSmith tracing when set to `true`.

`LANGCHAIN_API_KEY` contains your optional LangSmith API key.

`LANGCHAIN_PROJECT` controls the LangSmith project name. The default is `pdf-rag-streamlit-app`.

`LANGCHAIN_ENDPOINT` controls the LangSmith API endpoint. The default is `https://api.smith.langchain.com`.

No real key should be committed. `.env` is ignored by Git and should stay only on your local machine.

## LangSmith Traceability

LangSmith records and visualizes LangChain runs. It helps inspect retrieval, prompts, model calls, latency, and errors. To use it, create a LangSmith API key, place it in `.env`, and set `LANGCHAIN_TRACING_V2=true`.

`LANGCHAIN_PROJECT` controls the project name shown in the LangSmith dashboard. Tracing is optional, and the application works without LangSmith.

To disable tracing:

```env
LANGCHAIN_TRACING_V2=false
```

Retrieved PDF text, prompts, and model responses may appear in traces. Do not upload confidential documents unless your data-governance rules and LangSmith configuration permit it. API keys must never be placed in trace metadata.

## Usage instructions

1. Start the Streamlit application.
2. Open the local Streamlit URL.
3. Upload a text-based PDF.
4. Wait for processing to complete.
5. Enter a question.
6. Review the grounded answer.
7. Expand the source section to inspect supporting pages.
8. Use the reset button before switching workflows when needed.

## Security notes

- Never commit `.env`.
- Never share API keys.
- Rotate a key immediately if it is exposed.
- Avoid logging environment variables.
- OpenAI API requests may incur charges.
- LangSmith traces may contain retrieved PDF text.
- Confidential-document handling depends on your privacy requirements.

## Troubleshooting

`OPENAI_API_KEY` missing: Add it to `.env`, then restart Streamlit.

OpenAI authentication failure: Check that your OpenAI key is active, correctly copied, and has access to the configured models.

LangSmith authentication failure: Check `LANGCHAIN_API_KEY`, or set `LANGCHAIN_TRACING_V2=false` to run without tracing.

Tracing not appearing: Confirm `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY` is set, and `LANGCHAIN_PROJECT` is the project you are viewing.

FAISS installation problems: Upgrade pip, then reinstall with `pip install --upgrade faiss-cpu`.

PDF text not extracted: Confirm the file contains selectable text.

Scanned PDF requiring OCR: OCR is not included. Use a text-based PDF or add OCR as a future enhancement.

Corrupted or encrypted PDF: Upload an unlocked, valid PDF.

Empty document: Upload a PDF with extractable text.

Dependency conflicts: Create a fresh virtual environment and reinstall `requirements.txt`.

Streamlit command not found: Activate the virtual environment and reinstall dependencies.

Virtual environment not activated: Run the activation command for your shell before installing or starting the app.

Model-access error: Change `OPENAI_CHAT_MODEL` or `OPENAI_EMBEDDING_MODEL` to models available to your OpenAI account.

Rate-limit error: Wait and retry, or check your OpenAI usage limits.

## Known limitations

- Only one PDF is supported at a time.
- OCR is not included.
- Scanned image-only PDFs may not work.
- Large PDFs may require more processing time and API usage.
- Embeddings are generated through OpenAI and may incur charges.
- The FAISS database is kept in memory.
- Restarting the application clears the vector store.
- Chat history is limited to the current Streamlit session.
- Answers depend on the quality of extracted text and retrieved chunks.

## Future enhancements

- Multiple PDF uploads
- OCR support
- Persistent vector databases
- Chroma or PostgreSQL with pgvector
- Hybrid semantic and keyword search
- Reranking
- Metadata filtering
- User authentication
- Streaming responses
- Conversation-aware question rewriting
- Docker support
- Automated tests
- Evaluation datasets
- LangSmith RAG evaluation
- Feedback collection
- Cloud deployment
