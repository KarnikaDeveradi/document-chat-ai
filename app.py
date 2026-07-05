import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIR = "chroma-db"

st.set_page_config(page_title="RAG Chat", page_icon="📄", layout="wide")


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say: I could not find the answer in the document.""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


def process_pdf(uploaded_file):
    """Save the uploaded PDF, chunk it, and (re)build the Chroma vectorstore.

    The old vectorstore is wiped first so documents from a previous upload
    don't get mixed in with the new one.
    """
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embedding_model = get_embedding_model()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
        )
    finally:
        os.remove(tmp_path)

    return vectorstore, len(chunks)


def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


def answer_question(retriever, llm, question):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)
    return response.content, docs


# ---------------- Session state ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ---------------- Sidebar: upload ----------------
with st.sidebar:
    st.header("📄 Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process document", type="primary"):
            with st.spinner("Reading, chunking, and embedding your PDF..."):
                vectorstore, n_chunks = process_pdf(uploaded_file)
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_name = uploaded_file.name
            st.session_state.messages = []
            st.success(f"Processed '{uploaded_file.name}' into {n_chunks} chunks.")

    if st.session_state.doc_name:
        st.info(f"Active document: **{st.session_state.doc_name}**")

    if st.button("Clear database"):
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        st.session_state.vectorstore = None
        st.session_state.doc_name = None
        st.session_state.messages = []
        st.success("Cleared.")

# ---------------- Main: chat ----------------
st.title("🧠 Chat with your PDF")

if st.session_state.vectorstore is None:
    st.warning("Upload and process a PDF from the sidebar to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        retriever = get_retriever(st.session_state.vectorstore)
        llm = get_llm()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = answer_question(retriever, llm, question)
                st.markdown(answer)
                with st.expander("Sources"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Chunk {i}** (page {d.metadata.get('page', '?')})")
                        st.text(d.page_content[:500])

        st.session_state.messages.append({"role": "assistant", "content": answer})