# # ------------------- Automatic Installer for Required Packages -------------------
# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# packages = {
#     "streamlit": "streamlit",
#     "langchain": "langchain==0.1.14",
#     "google.generativeai": "google-generativeai==0.3.2",
#     "tiktoken": "tiktoken",
#     "unstructured": "unstructured",
#     "pdf2image": "pdf2image",
#     "matplotlib": "matplotlib",
#     "plotly": "plotly",
#     "sklearn": "scikit-learn",
#     "pdfplumber": "pdfplumber",
#     "fitz": "PyMuPDF",
#     "openai": "openai",
#     "pandas": "pandas",
#     "tqdm": "tqdm",
#     "seaborn": "seaborn",
#     "sentence_transformers": "sentence-transformers",
#     "unstructured_inference": "unstructured-inference"
# }

# for module_name, pip_name in packages.items():
#     try:
#         __import__(module_name)
#     except ImportError:
#         install(pip_name)

# # --------------------- Imports (After Ensuring Installation) ---------------------
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PyPDF2 import PdfReader
from unstructured.partition.pdf import partition_pdf
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Dict
import google.generativeai as genai
from io import BytesIO
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="Financial PDF Q&A", layout="wide")
st.title("📊 Financial Report RAG Assistant")

# ------------------- Session State Initialization -------------------
for key in ["all_chunks", "all_elements", "all_tables", "embedding_model", "db"]:
    if key not in st.session_state:
        st.session_state[key] = []

# ------------------- Sidebar Configuration -------------------
st.sidebar.header("🔧 Configuration")
api_key = st.sidebar.text_input("Enter your API key:", type="password")
llm_choice = st.sidebar.selectbox("Choose LLM", ["gemini-1.5-flash", "gemini-2.0-flash"])

# ------------------- File Upload -------------------
st.subheader("📄 Upload PDF Files (up to 3)")
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    pdf_files = uploaded_files[:3]
    st.write("Uploaded files:")
    for pdf_file in pdf_files:
        st.write(pdf_file.name)
    st.success(f"Successfully uploaded {len(pdf_files)} PDF files.")
else:
    st.info("Please upload up to 3 PDF files.")

# ------------------- Query Input -------------------
st.subheader("❓ Ask a Question")
user_query = st.text_input("Enter your query about the uploaded files")

# ------------------- Placeholders -------------------
response_placeholder = st.empty()
plot_placeholder = st.empty()

# ------------------- Utility Functions -------------------
def get_pdf_page_count(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        return len(reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return 0

def chunk_pages(total_pages, chunk_size=20):
    return [(start, min(start + chunk_size - 1, total_pages)) for start in range(1, total_pages + 1, chunk_size)]

def extract_mixed_content(file_bytes_data, page_range):
    try:
        start, end = page_range
        pages = f"{start}-{end}"
        if isinstance(file_bytes_data, bytes):
            file_bytes_data = BytesIO(file_bytes_data)
        elements = partition_pdf(
            file=file_bytes_data,
            strategy="fast",
            infer_table_structure=True,
            pages=pages,
            extract_images=True,
            include_page_breaks=True
        )
        return elements
    except Exception as e:
        print(f"[extract_mixed_content] ERROR on pages {page_range}: {e}")
        return []

def process_pdf_chunks(uploaded_file, page_chunks):
    raw_bytes = uploaded_file.getvalue()
    args = [(raw_bytes, chunk) for chunk in page_chunks]
    results = [extract_mixed_content(*arg) for arg in args]
    return [el for result in results for el in result]

def extract_tables_from_pdf(uploaded_file):
    try:
        import pdfplumber
        tables = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                extracted = page.extract_tables()
                for table in extracted:
                    table_str = f"Table on Page {page_num}:\n"
                    table_str += "\n".join(["\t".join([str(cell) if cell else "" for cell in row]) for row in table])
                    tables.append(table_str)
        return tables
    except Exception as e:
        print(f"[pdfplumber] Error extracting tables: {e}")
        return []

def is_text_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = reader.pages[0].extract_text()
        return bool(text and text.strip())
    except:
        return False

def chunk_texts(elements, chunk_size=10000, chunk_overlap=1000):
    texts = [str(e.text).strip() for e in elements if hasattr(e, 'text') and e.text and str(e.text).strip()]
    st.write(f"✅ Non-empty text elements found: {len(texts)}")
    if not texts:
        st.warning("⚠️ PDF might be scanned or image-based. No extractable text found.")
        return []
    full_text = "\n\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(full_text)

def tag_chunks(chunks: List[str], source_name: str) -> List[Dict]:
    return [{"text": chunk, "metadata": {"source": source_name}} for chunk in chunks]

# ------------------- Action Buttons -------------------
col1, col2, col3 = st.columns(3)
with col1:
    process_btn = st.button("🔍 Get Response")
with col2:
    visualize_chunks_btn = st.button("📈 2D Chunk Distribution")
with col3:
    visualize_embeddings_btn = st.button("🌐 3D Document Embedding")

# ------------------- Main Logic -------------------
if process_btn and uploaded_files and user_query:
    st.session_state.all_chunks = []
    st.session_state.all_elements = []
    st.session_state.all_tables = []

    for pdf in pdf_files:
        if not is_text_pdf(pdf):
            st.warning(f"⚠️ The PDF '{pdf.name}' appears to be scanned or image-based. Consider using OCR tools.")
            continue

        total_pages = get_pdf_page_count(pdf)
        page_chunks = chunk_pages(total_pages)
        elements = process_pdf_chunks(pdf, page_chunks)
        st.session_state.all_elements.extend(elements)

        tables = extract_tables_from_pdf(pdf)
        st.session_state.all_tables.extend(tables)

        chunks = chunk_texts(elements)
        tagged_chunks = tag_chunks(chunks, pdf.name)
        st.session_state.all_chunks.extend(tagged_chunks)

    st.success("PDF Processing Complete")
    st.write(f"🔹 Chunks: {len(st.session_state.all_chunks)} | Tables: {len(st.session_state.all_tables)} | Elements: {len(st.session_state.all_elements)}")

    converted_chunks = [
        Document(page_content=ch['text'], metadata=ch['metadata'])
        for ch in st.session_state.all_chunks if ch['text'].strip()
    ]

    if not converted_chunks:
        st.error("❗ No valid text chunks found to create the FAISS index.")
    else:
        st.session_state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.db = FAISS.from_documents(converted_chunks, st.session_state.embedding_model)
        st.session_state.db.save_local("faiss_store")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(llm_choice)

        def retrieve_relevant_chunks(query, db, top_k=5):
            return [doc.page_content for doc in db.similarity_search(query, k=top_k)]

        prompt_template = '''
You are a highly skilled and experienced financial analyst AI assistant.

You are given extracted content from financial reports. The content may include descriptions, figures, and also text representations of tables (where column alignment might not be perfect).

Context:
{context}

Your task is to:
1. Carefully analyze the numbers and tabular-like structures.
2. If tabular data is found, treat repeated values as rows and align them mentally with column headers.
3. Think step-by-step like a financial analyst while calculating or interpreting figures.

Now, based on the above context, answer the following user question with detailed reasoning and Present the information in a well-organized and aesthetically pleasing format that is
easy for a user to understand.

"{query}"

If the information is not sufficient to answer, say so clearly. Don't Hallucinate at all.
'''

        retrieved = retrieve_relevant_chunks(user_query, st.session_state.db)
        prompt = prompt_template.format(context="\n\n".join(retrieved), query=user_query)
        response = model.generate_content(prompt)
        response_placeholder.markdown(f"### 💡 Answer:\n\n{response.text}")

# ------------------- 2D Chunk Distribution -------------------
if visualize_chunks_btn and st.session_state.all_chunks:
    all_texts = [chunk['text'] for chunk in st.session_state.all_chunks]
    chunk_lengths = [len(text) for text in all_texts]
    if chunk_lengths:
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=100)
        ax.hist(chunk_lengths, bins=30, color='skyblue', edgecolor='black')
        ax.set_title("📊 Chunk Length Distribution (All Chunks)")
        ax.set_xlabel("Length (characters)")
        ax.set_ylabel("Number of Chunks")
        ax.grid(True)
        plot_placeholder.pyplot(fig)
        st.info(f"Total chunks visualized: {len(chunk_lengths)}")
    else:
        st.warning("⚠️ No chunks found for 2D visualization.")

# ------------------- 3D Embedding Visualization -------------------
if visualize_embeddings_btn and st.session_state.all_chunks:
    converted_chunks = [
        Document(page_content=chunk['text'], metadata=chunk['metadata'])
        for chunk in st.session_state.all_chunks
    ]
    all_texts = [doc.page_content for doc in converted_chunks]

    if not st.session_state.embedding_model:
        st.warning("⚠️ Embedding model not found. Please run 'Get Response' first.")
    else:
        all_vectors = np.array(st.session_state.embedding_model.embed_documents(all_texts))

        if all_vectors.shape[0] >= 3:
            pca = PCA(n_components=3)
            reduced = pca.fit_transform(all_vectors)
            fig = plt.figure(figsize=(4.5, 4.5), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], alpha=0.7, c='blue')
            ax.set_title("🔍 3D Document Embedding PCA")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.set_zlabel("PC 3")
            plot_placeholder.pyplot(fig)
        else:
            st.warning("⚠️ Not enough data for 3D PCA visualization. Need at least 3 chunks.")
