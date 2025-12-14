import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📄 PDF Question Answering using RAG")

# Upload PDF
pdf_file = st.file_uploader("Upload your pdf", type="pdf")

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"   # or gpt-4o-mini
    )

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Ask Question
    query = st.text_input("Ask a question from the PDF")

    if query:
        result = qa_chain(query)
        st.subheader("Answer")
        st.write(result["result"])

        with st.expander("📌 Source Chunks"):
            for doc in result["source_documents"]:
                st.write(doc.page_content)

    os.remove(pdf_path)
