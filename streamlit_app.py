import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import os
import json

st.set_page_config(
    page_title="FREE Internal Docs Q&A", 
    page_icon="📚",
    layout="wide"
)

def initialize_app():
    """Initialize the app"""
    if 'initialized' not in st.session_state:
        with st.spinner("🚀 Initializing FREE AI..."):
            try:
                # Configure FREE Google Gemini
                genai.configure(api_key=st.secrets["AIzaSyDATBQKVZQFNo8j9aLXz6e105bZQ3VikC4"])
                st.session_state.model = genai.GenerativeModel('gemini-pro')
                st.session_state.documents = []
                st.session_state.initialized = True
                st.success("✅ Ready to use!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()

def calculate_similarity(text1, text2):
    """Simple text similarity using word overlap"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0

def main():
    st.title("📚 100% FREE Internal Docs Q&A Agent")
    st.markdown("### 💰 Zero cost • 🚀 No SQLite issues!")
    
    # Initialize
    initialize_app()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            help="Completely FREE - No database needed!"
        )
        
        if uploaded_file and st.button("🚀 Process Document"):
            process_document(uploaded_file)
        
        st.markdown("---")
        
        # Stats
        doc_count = len(st.session_state.documents)
        st.metric("📊 Documents Processed", doc_count)
        
        if st.button("🗑️ Clear All Documents"):
            st.session_state.documents = []
            st.success("Cleared all documents!")
            st.rerun()
            
        st.markdown("### 💡 Sample Questions")
        st.markdown("""
        - What's our vacation policy?
        - How do I submit expenses? 
        - What are the security rules?
        - Where is the style guide?
        """)
        
        st.markdown("---")
        st.markdown("### ✅ No Database Issues!")
        st.markdown("Uses simple in-memory storage")

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
            with st.spinner("🧠 Searching documents..."):
                response = get_answer(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def process_document(uploaded_file):
    """Process uploaded document - No database needed"""
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
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separator="\n"
            )
            chunks = text_splitter.split_documents(docs)
            
            # Store chunks in memory (no database!)
            for i, chunk in enumerate(chunks):
                doc_data = {
                    'text': chunk.page_content,
                    'source': uploaded_file.name,
                    'chunk_id': i + 1
                }
                st.session_state.documents.append(doc_data)
            
            os.unlink(tmp_file_path)
            
            st.success(f"✅ Processed {uploaded_file.name}")
            st.info(f"📊 Created {len(chunks)} text chunks")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error processing document: {e}")

def get_answer(question):
    """Get answer using simple text search"""
    try:
        # Check if documents exist
        if not st.session_state.documents:
            return "❌ Please upload some documents first!"
        
        # Find most relevant chunks using simple similarity
        scored_docs = []
        for doc in st.session_state.documents:
            similarity = calculate_similarity(question, doc['text'])
            scored_docs.append((similarity, doc))
        
        # Sort by similarity and get top 3
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        top_docs = scored_docs[:3]
        
        # Filter out very low similarity scores
        relevant_docs = [(score, doc) for score, doc in top_docs if score > 0.1]
        
        if not relevant_docs:
            return "🤔 I couldn't find relevant information in your documents for this question."
        
        # Create context from relevant documents
        context_parts = []
        sources = []
        
        for score, doc in relevant_docs:
            context_parts.append(doc['text'])
            sources.append(f"📄 {doc['source']} (chunk {doc['chunk_id']})")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create prompt for Gemini
        prompt = f"""Based ONLY on the following document excerpts, please answer the user's question. If the answer cannot be found in the documents, say so clearly.

DOCUMENT EXCERPTS:
{context}

USER QUESTION: {question}

Please provide a helpful and accurate answer based only on the information provided above. If you cannot find the answer in these documents, please say "I cannot find information about this in the uploaded documents."

ANSWER:"""

        # Get answer from Gemini
        response = st.session_state.model.generate_content(prompt)
        answer = response.text
        
        # Add sources
        unique_sources = list(set(sources))
        sources_text = "\n\n**📚 Sources:**\n" + "\n".join(f"• {source}" for source in unique_sources[:3])
        
        return answer + sources_text
        
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

def show_debug_info():
    """Show debug information"""
    with st.expander("🔍 Debug Info"):
        st.write(f"Documents loaded: {len(st.session_state.documents)}")
        st.write("Document sources:")
        sources = set()
        for doc in st.session_state.documents:
            sources.add(doc['source'])
        for source in sources:
            st.write(f"• {source}")

if __name__ == "__main__":
    main()
    show_debug_info()
