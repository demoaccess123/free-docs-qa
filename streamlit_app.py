import streamlit as st
import google.generativeai as genai
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
    st.session_state.documents = []

def initialize_models():
    """Initialize FREE models"""
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing FREE AI models..."):
            try:
                # Configure FREE Google Gemini
                genai.configure(api_key=st.secrets["AIzaSyDATBQKVZQFNo8j9aLXz6e105bZQ3VikC4"])
                st.session_state.model = genai.GenerativeModel('gemini-pro')
                
                # Initialize FREE ChromaDB (simple version)
                st.session_state.chroma_client = chromadb.Client()
                
                try:
                    st.session_state.collection = st.session_state.chroma_client.get_collection("docs")
                except:
                    st.session_state.collection = st.session_state.chroma_client.create_collection("docs")
                
                st.session_state.initialized = True
                st.success("✅ FREE AI ready to use!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

def simple_embedding(text):
    """Simple text embedding using character frequency"""
    # Create a simple numerical representation of text
    chars = "abcdefghijklmnopqrstuvwxyz "
    vector = []
    text = text.lower()
    
    for char in chars:
        count = text.count(char)
        vector.append(count / len(text) if len(text) > 0 else 0)
    
    # Pad or trim to consistent length
    while len(vector) < 100:
        vector.append(0.0)
    
    return vector[:100]

def main():
    st.title("📚 100% FREE Internal Docs Q&A Agent")
    st.markdown("### 💰 Zero cost • 🚀 Powered by Google Gemini")
    
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
        doc_count = len(st.session_state.documents)
        st.metric("📊 Documents Processed", doc_count)
            
        st.markdown("### 💡 Sample Questions")
        st.markdown("""
        - What's our vacation policy?
        - How do I submit expenses?
        - What are the security rules?
        - Where is the style guide?
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
    """Process uploaded document"""
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
            
            # Store in session state (simple approach)
            for i, chunk in enumerate(chunks):
                doc_data = {
                    'text': chunk.page_content,
                    'source': uploaded_file.name,
                    'chunk': i,
                    'embedding': simple_embedding(chunk.page_content)
                }
                st.session_state.documents.append(doc_data)
            
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
        if not st.session_state.documents:
            return "❌ Please upload some documents first!"
        
        # Simple similarity search
        question_embedding = simple_embedding(question)
        
        # Find most similar documents (simple approach)
        similarities = []
        for doc in st.session_state.documents:
            # Simple cosine similarity
            similarity = sum(a * b for a, b in zip(question_embedding, doc['embedding']))
            similarities.append((similarity, doc))
        
        # Get top 3 most similar
        similarities.sort(reverse=True)
        top_docs = similarities[:3]
        
        if not top_docs:
            return "🤔 No relevant information found."
        
        # Create context
        context = "\n\n".join([doc['text'] for _, doc in top_docs])
        sources = [f"📄 {doc['source']}" for _, doc in top_docs]
        
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
