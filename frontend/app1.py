"""
Fire Protection RAG System - Professional Chat Interface
Clean design, subtle colors, smart source handling
"""
import streamlit as st
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import re

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from backend.extractors.pdf_text_extractor import PDFTextExtractor
from backend.processors.chunkers import SmartDocumentChunker

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Bms-Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        background: #f8f9fa;
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* User message */
    .user-message {
        background: #e3f2fd;
        color: #1565c0;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0;
        max-width: 70%;
        margin-left: auto;
        border: 1px solid #bbdefb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        animation: slideIn 0.3s ease;
    }
    
    /* Assistant message */
    .assistant-message {
        background: white;
        color: #2c3e50;
        padding: 1rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem 0;
        max-width: 75%;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        animation: slideIn 0.3s ease;
    }
    
    .assistant-message p {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    .assistant-message strong {
        color: #1565c0;
    }
    
    .assistant-message ul, .assistant-message ol {
        margin: 0.8rem 0;
        padding-left: 1.5rem;
    }
    
    .assistant-message li {
        margin: 0.4rem 0;
    }
    
    /* System message */
    .system-message {
        background: #f5f5f5;
        color: #555;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #2c3e50;
        font-size: 0.95rem;
    }
    
    /* Source section (only shown when requested) */
    .sources-section {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .sources-header {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
    }
    
    .source-item {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.6rem 0;
        font-size: 0.9rem;
    }
    
    .source-meta {
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }
    
    .source-preview {
        color: #444;
        font-style: italic;
        line-height: 1.5;
        margin-top: 0.4rem;
        padding-left: 0.8rem;
        border-left: 2px solid #ddd;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2c3e50;
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

if 'pinecone' not in st.session_state:
    st.session_state.pinecone = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'llm' not in st.session_state:
    st.session_state.llm = None

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'namespace' not in st.session_state:
    st.session_state.namespace = None

if 'stats' not in st.session_state:
    st.session_state.stats = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_greeting(text: str) -> bool:
    """Check if message is a greeting"""
    greetings = [
        r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
        r'\b(howdy|yo|sup|whats up|what\'s up)\b',
        r'^(hii+|helloo+|heyy+)$'
    ]
    text_lower = text.lower().strip()
    return any(re.search(pattern, text_lower) for pattern in greetings)

def is_asking_for_sources(text: str) -> bool:
    """Check if user is asking for source citations"""
    source_keywords = [
        r'\b(source|sources|reference|references|citation|citations)\b',
        r'\b(where.*from|which page|what page|page number)\b',
        r'\b(show.*source|provide.*source|give.*source)\b',
    ]
    text_lower = text.lower().strip()
    return any(re.search(pattern, text_lower) for pattern in source_keywords)

def get_greeting_response() -> str:
    """Generate friendly greeting response"""
    return """Hello! I'm your AI Assistant.

I specialize in helping with fire protection documentation and standards. I can:
• Answer questions about NFPA standards and requirements
• Explain technical specifications and formulas
• Summarize procedures and guidelines
• Find specific information in your documentation

What would you like to know?"""

def clean_text(text: str) -> str:
    """Clean extracted text from PDF artifacts"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove weird encoding artifacts
    text = re.sub(r'[^\w\s\.,;:!?\-()\'\"\/]', '', text)
    # Remove copyright notices
    text = re.sub(r'Copyright.*?(?:\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Copyrighted material.*?(?:\.|$)', '', text, flags=re.IGNORECASE)
    # Clean up spacing
    text = text.strip()
    return text

# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================
def initialize_system():
    """Initialize all components"""
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "bms-agent")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        
        if not openai_key or not pinecone_key:
            return False, "Missing API keys in .env file"
        
        if not st.session_state.pinecone:
            st.session_state.pinecone = Pinecone(api_key=pinecone_key)
        
        index = st.session_state.pinecone.Index(index_name)
        stats = index.describe_index_stats()
        
        namespaces = stats.get('namespaces', {})
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors == 0:
            return False, "No documents found. Please upload PDFs first."
        
        if namespaces and not st.session_state.namespace:
            best_ns = max(namespaces.items(), key=lambda x: x[1].get('vector_count', 0))
            st.session_state.namespace = best_ns[0]
        
        if not st.session_state.embeddings:
            st.session_state.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=openai_key
            )
        
        if not st.session_state.llm:
            st.session_state.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                openai_api_key=openai_key
            )
        
        if not st.session_state.vector_store:
            st.session_state.vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=st.session_state.embeddings,
                namespace=st.session_state.namespace
            )
        
        st.session_state.stats = {
            'total_vectors': total_vectors,
            'namespaces': namespaces,
            'index_name': index_name,
            'current_namespace': st.session_state.namespace
        }
        
        st.session_state.system_ready = True
        return True, "System ready"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

# ============================================================================
# RAG QUERY
# ============================================================================
def get_rag_answer(question: str, threshold: float = 0.3, top_k: int = 5, include_sources: bool = False):
    """Get answer using RAG"""
    try:
        # Retrieve documents
        results = st.session_state.vector_store.similarity_search_with_score(
            query=question,
            k=top_k
        )
        
        if not results:
            return {
                'answer': "I couldn't find relevant information in the documentation. Could you rephrase your question or ask about a different topic?",
                'sources': [],
                'has_results': False
            }
        
        # Filter by threshold
        filtered = [(doc, score) for doc, score in results if score >= threshold]
        
        if not filtered:
            best_score = results[0][1]
            return {
                'answer': f"I found some content but it wasn't relevant enough (best match: {best_score:.2f}). Try lowering the similarity threshold in the sidebar.",
                'sources': [],
                'has_results': False
            }
        
        # Format context
        context_parts = []
        sources = []
        
        for idx, (doc, score) in enumerate(filtered[:5], 1):
            page = doc.metadata.get('page', 'Unknown')
            section = doc.metadata.get('section', 'Unknown')
            
            # Clean the text
            clean_content = clean_text(doc.page_content)
            
            context_parts.append(
                f"[Document {idx} - Page {page}, Section {section}]\n{clean_content}"
            )
            
            # Only store sources if needed
            if include_sources and len(clean_content) > 50:
                sources.append({
                    'page': page,
                    'section': section,
                    'score': score,
                    'content': clean_content[:300]  # First 300 chars
                })
        
        context = "\n\n".join(context_parts)
        
        # Create helpful prompt
        prompt = f"""You are an expert fire protection engineering assistant.

CONTEXT FROM DOCUMENTATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, helpful, and professional answer
- Synthesize information from the context
- Use bullet points or numbered lists when appropriate
- Be concise but comprehensive
- Explain technical terms clearly
- Do NOT mention page numbers or sources in your answer unless specifically asked

Provide your response:"""
        
        # Get answer from LLM
        response = st.session_state.llm.invoke(prompt)
        answer = response.content
        
        return {
            'answer': answer,
            'sources': sources if include_sources else [],
            'has_results': True
        }
        
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'sources': [],
            'has_results': False
        }

# ============================================================================
# MESSAGE DISPLAY
# ============================================================================
def display_message(role: str, content: str, sources=None, show_sources=False):
    """Display a message in chat format"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
        
    elif role == "assistant":
        # Convert markdown-style formatting to HTML
        content_html = content.replace('\n', '<br>')
        
        st.markdown(f"""
        <div class="assistant-message">
            {content_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources only if requested and available
        if show_sources and sources and len(sources) > 0:
            sources_html = '<div class="sources-section">'
            sources_html += '<div class="sources-header">📚 Source References</div>'
            
            for idx, source in enumerate(sources, 1):
                page = source['page']
                section = source['section'] if source['section'] != 'Unknown' else 'N/A'
                score = source['score']
                content = source['content']
                
                sources_html += f"""
                <div class="source-item">
                    <div class="source-meta">
                        <strong>Reference {idx}:</strong> Page {page}
                        {f", Section {section}" if section != 'N/A' else ""}
                        (Relevance: {score:.2f})
                    </div>
                    <div class="source-preview">
                        "{content}..."
                    </div>
                </div>
                """
            
            sources_html += '</div>'
            st.markdown(sources_html, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Ask your queries </h1>
        <p class="header-subtitle">Professional guidance based on documentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.system_ready:
        with st.spinner("Initializing system..."):
            success, message = initialize_system()
            if success:
                st.success("✅ " + message)
            else:
                st.error("❌ " + message)
                st.info("""
                **Setup Required:**
                1. Ensure .env file has OPENAI_API_KEY and PINECONE_API_KEY
                2. Set EMBEDDING_MODEL=text-embedding-ada-002
                3. Set dimension: int = 1536 in config.py
                """)
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        if st.session_state.stats:
            st.success(f"✅ Connected")
            st.info(f"📊 {st.session_state.stats['total_vectors']} vectors indexed")
            
            st.markdown("---")
            
            # Document Upload Section
            st.markdown("### 📄 Upload New Document")
            
            with st.expander("➕ Add PDF", expanded=False):
                uploaded_file = st.file_uploader(
                    "Choose a PDF file",
                    type=['pdf'],
                    help="Upload fire protection documentation"
                )
                
                if uploaded_file:
                    if st.button(" Process & Index", use_container_width=True):
                        with st.spinner("Processing PDF..."):
                            try:
                                # Extract text
                                extractor = PDFTextExtractor()
                                pdf_data = extractor.extract_from_uploaded_file(uploaded_file)
                                
                                # Create chunks
                                chunker = SmartDocumentChunker(
                                    chunk_size=1000,
                                    chunk_overlap=200
                                )
                                documents = chunker.chunk_document(pdf_data)
                                
                                # Create namespace from filename
                                namespace = uploaded_file.name.replace('.pdf', '').replace(' ', '_')
                                
                                # Add to Pinecone using existing vector store class
                                vector_store_temp = PineconeVectorStore.from_documents(
                                    documents=documents,
                                    embedding=st.session_state.embeddings,
                                    index_name=st.session_state.stats['index_name'],
                                    namespace=namespace
                                )
                                
                                st.success(f"✅ Indexed {len(documents)} chunks from {len(pdf_data['pages'])} pages")
                                st.info(f"📁 Namespace: `{namespace}`")
                                
                                # Refresh system
                                st.session_state.system_ready = False
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                                st.exception(e)
            
            st.markdown("---")
            
            # Document Selection
            st.markdown("### 📚 Available Documents")
            
            if st.session_state.stats.get('namespaces'):
                namespaces = st.session_state.stats['namespaces']
                
                # Create list of namespace options
                namespace_options = {}
                for ns_name, ns_info in namespaces.items():
                    count = ns_info.get('vector_count', 0)
                    namespace_options[f"{ns_name} ({count} chunks)"] = ns_name
                
                # Current selection
                current_display = None
                for display, ns in namespace_options.items():
                    if ns == st.session_state.namespace:
                        current_display = display
                        break
                
                # Dropdown to select namespace
                selected_display = st.selectbox(
                    "Select active document:",
                    options=list(namespace_options.keys()),
                    index=list(namespace_options.values()).index(st.session_state.namespace) if st.session_state.namespace in namespace_options.values() else 0,
                    help="Choose which document to search"
                )
                
                selected_namespace = namespace_options[selected_display]
                
                # If changed, update and refresh vector store
                if selected_namespace != st.session_state.namespace:
                    st.session_state.namespace = selected_namespace
                    
                    # Recreate vector store with new namespace
                    st.session_state.vector_store = PineconeVectorStore(
                        index_name=st.session_state.stats['index_name'],
                        embedding=st.session_state.embeddings,
                        namespace=selected_namespace
                    )
                    
                    st.success(f"✅ Switched to: {selected_namespace}")
                    st.rerun()
                
                # Show active document info
                st.info(f"🔍 Currently searching: **{st.session_state.namespace}**")
            
            st.markdown("---")
            st.markdown("### 🎛️ Settings")
            
            threshold = st.slider(
                "Similarity Threshold",
                0.0, 1.0, 0.3, 0.05,
                help="Lower = more results"
            )
            
            top_k = st.slider(
                "Documents to Retrieve",
                1, 10, 5,
                help="Number of document chunks"
            )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.session_state.system_ready = False
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### 💡 Quick Tips")
            st.markdown("""
            - Start with a greeting
            - Ask specific questions
            - Request summaries
            - Say "show sources" to see references
            """)
    
    # Main chat area
    st.markdown("### 💬 Conversation")
    
    # Display messages
    if not st.session_state.messages:
        st.markdown("""
        <div class="system-message">
            <strong>Welcome!</strong> I'm here to help with fire protection documentation.
            Start by asking a question or just say hello!
        </div>
        """, unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        display_message(
            msg['role'],
            msg['content'],
            msg.get('sources'),
            msg.get('show_sources', False)
        )
    
    # Chat input
    user_input = st.chat_input("Ask about fire protection systems...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input
        })
        
        # Determine if sources should be shown
        wants_sources = is_asking_for_sources(user_input)
        
        # Check if greeting
        if is_greeting(user_input):
            response = get_greeting_response()
            sources = []
            show_sources = False
        else:
            # Get RAG answer
            with st.spinner("Thinking..."):
                result = get_rag_answer(user_input, threshold, top_k, wants_sources)
                response = result['answer']
                sources = result['sources']
                show_sources = wants_sources
        
        # Add assistant message
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'sources': sources,
            'show_sources': show_sources
        })
        
        st.rerun()

if __name__ == "__main__":
    main()

