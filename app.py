import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import PyPDF2
from io import StringIO

# Initialize the Pinecone vector database
api_key = "pcsk_4UmAuP_TqfKF1c7jiAEB488axzUb6gxedvYHqwGTGcktmdiLgLjb4fYzritCYJn7pHrnJw"
region = "us-east-1"
pc = Pinecone(api_key=api_key)
index_name = "document-qa"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=region
        )
    )
index = pc.Index(index_name)

# Load the Hugging Face model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

def process_pdf(file):
    """Extract text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    """Chunk text into smaller pieces."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

def embed_text(chunks):
    """Generate vector embeddings for text chunks."""
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return embeddings

def store_embeddings(chunks, embeddings):
    """Store embeddings in Pinecone."""
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        index.upsert([(f"chunk-{i}", embedding.tolist(), {"text": chunk})])

def answer_question(question):
    """Answer user questions based on document embeddings."""
    question_embedding = model.encode(question, convert_to_tensor=True).tolist()
    # Update the query call with keyword arguments
    search_results = index.query(vector=question_embedding, top_k=3, include_metadata=True)
    answers = [result["metadata"]["text"] for result in search_results["matches"]]
    return answers


# Streamlit UI
st.set_page_config(page_title="Document QA System", layout="centered")
st.title("DocGPT")
st.markdown(
    """
    Upload your documents, and ask questions to get relevant answers.
    """
)

# Document upload
uploaded_file = st.file_uploader("Upload a PDF Document", type="pdf")
if uploaded_file:
    with st.spinner("Processing document..."):
        document_text = process_pdf(uploaded_file)
        chunks = list(chunk_text(document_text))
        embeddings = embed_text(chunks)
        store_embeddings(chunks, embeddings)
    st.success("Document processed and stored in the database.")

# Question-answering interface
question = st.text_input("Ask a question about the document")
if question:
    with st.spinner("Searching for answers..."):
        answers = answer_question(question)
    st.subheader("Answers")
    for answer in answers:
        st.write(answer)

# Styling
st.markdown(
    """
    <style>
    .stButton>button { background-color: #F5F5DC; color: #333; }
    .stTextInput>div>input { border: 1px solid #333; }
    body { background-color: #FAF3E0; color: #333; }
    </style>
    """,
    unsafe_allow_html=True
)