import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import os
import uuid

st.set_page_config(
    page_title="FREE Internal Docs Q&A", 
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_models():
    """Initialize all FREE models"""
    if not st.session_state.initialized:
        with st.spinner("🚀 Loading FREE AI models (first time only)..."):
            try:
                # Configure FREE Google Gemini
                genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                st.session_state.model = genai.GenerativeModel('gemini-pro')
                
                # Load FREE embedding model
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Initialize FREE ChromaDB
                st.session_state.chroma_client = chromadb.Client()
                
                try:
                    st.session_state.collection = st.session_state.chroma_client.get_collection("docs")
                except:
                    st.session_state.collection = st.session_state.chroma_client.create_collection("docs")
                
                st.session_state.initialized = True
                st.success("✅ FREE AI models loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error initializing: {e}")
                st.stop()

def main():
    st.title("📚 100% FREE Internal Docs Q&A Agent")
    st.markdown("### 💰 Zero cost • 🚀 Powered by Google Gemini & Open Source AI")
    
    # Initialize models
    initialize_models()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            help="Completely FREE document processing!"
        )
        
        if uploaded_file and st.button("🚀 Process Document"):
            process_document(uploaded_file)
        
        st.markdown("---")
        
        # Stats
        if st.session_state.initialized:
            doc_count = st.session_state.collection.count()
            st.metric("📊 Documents Processed", doc_count)
            
        st.markdown("### 💡 Sample Questions")
        st.markdown("""
        - What's our vacation policy?
        - How do I submit expenses?
        - What are the security rules?
        - Where is the style guide?
        """)
        
        st.markdown("---")
        st.markdown("### 🎉 100% FREE Stack")
        st.markdown("""
        - 🧠 **Google Gemini** (Free API)
        - 🔍 **SentenceTransformers** (Free)
        - 💾 **ChromaDB** (Free)
        - ☁️ **Streamlit Cloud** (Free)
        - 💰 **Total Cost: $0/month**
        """)

    # Main chat area
    st.header("💬 Ask Your Questions")
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                response = get_answer(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def process_document(uploaded_file):
    """Process uploaded document with FREE models"""
    try:
        with st.spinner("📖 Processing document..."):
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name
            
            # Load document
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path)
            
            docs = loader.load()
            
            # Split text
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            # Generate embeddings
            texts = [doc.page_content for doc in chunks]
            embeddings = st.session_state.embedding_model.encode(texts).tolist()
            
            # Store in ChromaDB
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            metadatas = [{"source": uploaded_file.name, "chunk": i} for i in range(len(texts))]
            
            st.session_state.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            os.unlink(tmp_file_path)
            
            st.success(f"✅ Processed {uploaded_file.name}")
            st.info(f"📊 Created {len(chunks)} chunks")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error: {e}")

def get_answer(question):
    """Get answer using FREE models"""
    try:
        # Check if documents exist
        if st.session_state.collection.count() == 0:
            return "❌ Please upload some documents first!"
        
        # Search documents
        question_embedding = st.session_state.embedding_model.encode([question]).tolist()[0]
        results = st.session_state.collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        
        if not results['documents']:
            return "🤔 No relevant information found in your documents."
        
        # Create context
        context = "\n\n".join(results['documents'])
        sources = [f"📄 {meta['source']}" for meta in results['metadatas']]
        
        # Ask Gemini
        prompt = f"""Based on these documents, answer the question:

DOCUMENTS:
{context}

QUESTION: {question}

Provide a helpful answer based only on the document information."""

        response = st.session_state.model.generate_content(prompt)
        answer = response.text
        
        # Add sources
        sources_text = "\n\n**📚 Sources:** " + " • ".join(set(sources))
        
        return answer + sources_text
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    main()
