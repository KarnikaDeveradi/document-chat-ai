# Document Chat AI

A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and have a conversation with it. Instead of scrolling through pages looking for an answer, you upload the document once and just ask questions — the app finds the most relevant sections and uses an LLM to answer strictly from that content.

This project was built as a hands-on exercise in RAG architecture: document ingestion, chunking, embedding, vector search, and grounded LLM responses, wrapped in a usable interface.

## Why RAG?

A general-purpose LLM doesn't know what's inside your specific PDF, and pasting an entire document into a prompt is often impractical (token limits, cost, and irrelevant context diluting the answer). RAG solves this by:

1. Breaking the document into small chunks
2. Converting each chunk into a vector embedding (a numeric representation of meaning)
3. Storing those embeddings in a vector database
4. At query time, retrieving only the chunks most relevant to the question
5. Passing just those chunks — not the whole document — to the LLM as context

This keeps answers grounded in the actual source material and avoids the LLM "making things up" (hallucinating) about content it was never given.

## Features

- Upload any PDF through the sidebar
- Automatically splits and embeds the document into a vector store (Chroma)
- Ask questions in a chat interface, with conversation history preserved for the session
- Answers are generated only from the document's content — if the answer isn't in the document, the app says so instead of guessing
- Expandable "Sources" section under each answer showing exactly which chunks (and page numbers) were used
- "Clear database" button to reset and start fresh with a new document
- Re-uploading a document automatically wipes the previous one's data, so results are never mixed between documents

## How It Works (Architecture)

```
PDF Upload
   │
   ▼
PyPDFLoader (extracts text page by page)
   │
   ▼
RecursiveCharacterTextSplitter (splits into ~1000-character chunks, 200-character overlap)
   │
   ▼
HuggingFace Embeddings (all-MiniLM-L6-v2) — converts each chunk into a vector
   │
   ▼
ChromaDB (stores chunks + vectors, persisted to disk)
   │
   ▼
User asks a question
   │
   ▼
MMR Retriever (Maximal Marginal Relevance) — fetches top relevant, diverse chunks
   │
   ▼
Prompt Template — injects retrieved chunks + question into a strict "context-only" system prompt
   │
   ▼
Mistral LLM (mistral-small-2506) — generates the answer
   │
   ▼
Answer + source chunks shown in the UI
```

**Why MMR retrieval?** Instead of just grabbing the top-k most similar chunks (which can be redundant if several chunks say nearly the same thing), MMR balances relevance with diversity, so the LLM gets a broader, less repetitive slice of the document to work with.

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| UI | **Streamlit** | File upload, chat interface, session state |
| Orchestration | **LangChain** | Ties together loading, splitting, retrieval, and prompting |
| PDF parsing | **PyPDFLoader** | Extracts text from PDF pages |
| Embeddings | **HuggingFace `all-MiniLM-L6-v2`** | Converts text chunks into vectors (runs locally, no API cost) |
| Vector store | **ChromaDB** | Stores and searches embedded chunks |
| LLM | **Mistral AI `mistral-small-2506`** | Generates answers from retrieved context |

## Project Structure

```
document-chat-ai/
├── app.py              # Streamlit UI — upload, process, and chat
├── create_db.py         # Standalone script version of the ingestion step (optional, for reference)
├── main.py               # Standalone terminal version of the chat loop (optional, for reference)
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes .env, chroma-db/, .venv/, etc.
└── README.md
```

Only `app.py` and `requirements.txt` are required to run the full experience — `create_db.py` and `main.py` are kept for reference since they were the original building blocks this app was assembled from.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/KarnikaDeveradi/document-chat-ai.git
   cd document-chat-ai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Mistral API key:
   ```
   MISTRAL_API_KEY=your_api_key_here
   ```

4. Run the app:
   ```
   streamlit run app.py
   ```

## How to Use

1. Open the app in your browser (locally at `http://localhost:8501`, or the public URL if deployed).
2. In the sidebar, click **"Browse files"** and select a PDF.
3. Click **"Process document"**. This will:
   - Wipe any previously stored document's data
   - Extract text from every page of the PDF
   - Split the text into overlapping chunks
   - Generate embeddings and store them in ChromaDB
4. Once processing finishes, a success message shows how many chunks were created.
5. Type a question in the chat box at the bottom and press Enter.
6. Expand **"Sources"** under any answer to see exactly which chunks (and page numbers) the answer was based on.
7. To start over with a different document, either upload a new PDF (it auto-replaces the old one) or click **"Clear database"**.

## Limitations

- **One document at a time.** Uploading a new PDF discards the previous one rather than adding to a growing knowledge base. This keeps retrieval focused but means you can't currently ask questions across multiple documents at once.
- **No conversation memory in retrieval.** Each question is retrieved independently — the app doesn't yet factor previous chat turns into what it searches for, so follow-up questions like "what about the second one?" may not resolve correctly.
- **Text-based PDFs only.** Scanned/image-only PDFs won't extract usable text unless OCR is added separately.
- **Answer quality depends on chunking.** Very short or fragmented answers can occur if relevant information spans a chunk boundary; tuning `chunk_size`/`chunk_overlap` in `app.py` can help.

## Possible Future Improvements

- Support multiple documents in the same session, with the ability to filter or tag which document a question targets
- Add conversational memory so follow-up questions resolve correctly
- Support additional file types (`.docx`, `.txt`, `.md`)
- Add OCR fallback for scanned PDFs
- Cache embeddings so re-processing the same file doesn't redo work
