import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.chains import RetrievalQA


# 1. LOAD ENVIRONMENT VARIABLES FIRST

load_dotenv()

st.set_page_config(page_title="PDF RAG App ", layout="wide")
st.title("📄 PDF QUESTION ANSWERING SYSTEM")

# 2. CHECK FOR API KEY

if "GOOGLE_API_KEY" not in os.environ:
    st.error("The GOOGLE_API_KEY environment variable is not set.")
    st.info("Please create a `.env` file in your project root with GOOGLE_API_KEY=\"YOUR_KEY\" or set it as an environment variable.")
    st.stop()


# 3. STREAMLIT APPLICATION LOGIC

# Upload PDF
pdf_file = st.file_uploader("Upload your pdf", type="pdf")

if pdf_file:
    # 1. Save PDF to temporary file
    # We define pdf_path outside the try block so finally can access it
    pdf_path = None 
    
    try:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                pdf_path = tmp.name

            # 2. Load and Split text
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )
            docs = splitter.split_documents(documents)

            # 3. Embeddings (Automatically uses the loaded GOOGLE_API_KEY)
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004" 
            )

            # 4. Vector Store
            vectorstore = FAISS.from_documents(docs, embeddings)

            # 5. LLM (Automatically uses the loaded GOOGLE_API_KEY)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                temperature=0,
            )

            # 6. RAG Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
        
        st.success("Document processed successfully! Ready to ask questions.")

        # 7. Ask Question
        query = st.text_input("Ask a question from the PDF")

        if query:
            with st.spinner("Generating answer..."):
                result = qa_chain.invoke({"query": query})
                st.subheader("Answer")
                st.write(result["result"])

                with st.expander("📌 Source Chunks"):
                    for doc in result["source_documents"]:
                        # Ensure metadata exists before accessing it
                        page_number = doc.metadata.get('page')
                        # Note: PDF page numbers are often 0-indexed, so we add 1 for user readability
                        st.write(f"**Source Page:** {page_number + 1 if page_number is not None else 'N/A'}")
                        st.code(doc.page_content, language="markdown")
                        
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        # Optionally show a traceback for debugging
        # st.exception(e)

    finally:
        # 8. Cleanup temporary file - ENSURES deletion even if errors occur
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
            # st.info(f"Cleaned up temporary file: {pdf_path}") # Optional for debugging