"""
Comprehensive Diagnostic Tool
Tests: OpenAI API, Pinecone, Embeddings, Retrieval
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
sys.path.append(str(Path(__file__).parent))

load_dotenv()

print("="*70)
print("🔍 FIRE PROTECTION RAG SYSTEM - DIAGNOSTIC TOOL")
print("="*70)
print()

# ============================================================================
# TEST 1: Environment Variables
# ============================================================================
print("TEST 1: Checking Environment Variables")
print("-"*70)

openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "fire-extinguisher-docs")

if openai_key:
    print(f"✅ OPENAI_API_KEY: {openai_key[:20]}...{openai_key[-4:]}")
else:
    print("❌ OPENAI_API_KEY: NOT SET")

if pinecone_key:
    print(f"✅ PINECONE_API_KEY: {pinecone_key[:20]}...{pinecone_key[-4:]}")
else:
    print("❌ PINECONE_API_KEY: NOT SET")

print(f"📌 PINECONE_INDEX_NAME: {index_name}")
print()

if not openai_key or not pinecone_key:
    print("❌ Missing API keys! Fix your .env file first.")
    sys.exit(1)

# ============================================================================
# TEST 2: OpenAI API Connection
# ============================================================================
print("TEST 2: Testing OpenAI API Connection")
print("-"*70)

try:
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    
    # Test with a simple completion
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'API works!'"}],
        max_tokens=10
    )
    
    answer = response.choices[0].message.content
    print(f"✅ OpenAI API Working!")
    print(f"   Test response: {answer}")
    print()
    
except Exception as e:
    print(f"❌ OpenAI API Failed: {e}")
    print("   Check your OPENAI_API_KEY")
    print()
    sys.exit(1)

# ============================================================================
# TEST 3: OpenAI Embeddings
# ============================================================================
print("TEST 3: Testing OpenAI Embeddings")
print("-"*70)

try:
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key
    )
    
    # Test embedding
    test_text = "gas clean agent fire suppression system"
    embedding = embeddings.embed_query(test_text)
    
    print(f"✅ Embeddings Working!")
    print(f"   Test text: '{test_text}'")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    print()
    
except Exception as e:
    print(f"❌ Embeddings Failed: {e}")
    print()
    sys.exit(1)

# ============================================================================
# TEST 4: Pinecone Connection
# ============================================================================
print("TEST 4: Testing Pinecone Connection")
print("-"*70)

try:
    from pinecone import Pinecone
    
    pc = Pinecone(api_key=pinecone_key)
    
    # List indexes
    indexes = pc.list_indexes()
    index_names = [idx['name'] for idx in indexes]
    
    print(f"✅ Pinecone Connected!")
    print(f"   Available indexes: {index_names}")
    
    if index_name in index_names:
        print(f"   ✅ Target index '{index_name}' exists")
    else:
        print(f"   ❌ Target index '{index_name}' NOT FOUND")
        print(f"   Create it or update PINECONE_INDEX_NAME in .env")
        sys.exit(1)
    
    print()
    
except Exception as e:
    print(f"❌ Pinecone Connection Failed: {e}")
    print()
    sys.exit(1)

# ============================================================================
# TEST 5: Pinecone Index Stats
# ============================================================================
print("TEST 5: Checking Pinecone Index Contents")
print("-"*70)

try:
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    total_vectors = stats.get('total_vector_count', 0)
    namespaces = stats.get('namespaces', {})
    
    print(f"📊 Total vectors in index: {total_vectors}")
    
    if total_vectors == 0:
        print("❌ NO VECTORS IN INDEX!")
        print("   You need to upload and process PDFs first.")
        print()
        sys.exit(1)
    
    print(f"📁 Namespaces:")
    for ns_name, ns_info in namespaces.items():
        count = ns_info.get('vector_count', 0)
        print(f"   - '{ns_name}': {count} vectors")
    
    # Store namespace with most vectors for testing
    if namespaces:
        largest_ns = max(namespaces.items(), key=lambda x: x[1].get('vector_count', 0))
        test_namespace = largest_ns[0]
        print(f"\n   ✅ Will use namespace '{test_namespace}' for testing ({largest_ns[1]['vector_count']} vectors)")
    else:
        test_namespace = ""
        print(f"\n   ⚠️  No named namespaces found, using default")
    
    print()
    
except Exception as e:
    print(f"❌ Index Stats Failed: {e}")
    print()
    sys.exit(1)

# ============================================================================
# TEST 6: Actual Retrieval Test
# ============================================================================
print("TEST 6: Testing Document Retrieval")
print("-"*70)

try:
    from langchain_pinecone import PineconeVectorStore
    from langchain_core.documents import Document
    
    # Create vector store
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=test_namespace
    )
    
    # Test queries
    test_queries = [
        "gas clean agent",
        "fire suppression",
        "safety requirements",
        "what is this document about"
    ]
    
    print("Testing different queries with various thresholds:\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        
        # Try with different thresholds
        for threshold in [0.0, 0.3, 0.5, 0.7]:
            try:
                results = vector_store.similarity_search_with_score(
                    query=query,
                    k=3
                )
                
                # Filter by threshold
                filtered = [(doc, score) for doc, score in results if score >= threshold]
                
                if filtered:
                    best_score = filtered[0][1]
                    print(f"  Threshold {threshold}: ✅ Found {len(filtered)} docs (best score: {best_score:.3f})")
                else:
                    print(f"  Threshold {threshold}: ❌ No results")
                    
            except Exception as e:
                print(f"  Threshold {threshold}: ❌ Error: {e}")
        
        print()
    
    # Detailed look at best match
    print("="*70)
    print("DETAILED RESULTS for: 'gas clean agent'")
    print("="*70)
    
    results = vector_store.similarity_search_with_score(
        query="gas clean agent",
        k=5
    )
    
    if results:
        for idx, (doc, score) in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(f"  Score: {score:.3f}")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}")
            print(f"  Section: {doc.metadata.get('section', 'N/A')}")
            print(f"  Content preview: {doc.page_content[:200]}...")
    else:
        print("❌ NO RESULTS FOUND")
        print("This means your documents might not contain this information,")
        print("OR there's an issue with how they were indexed.")
    
    print()
    
except Exception as e:
    print(f"❌ Retrieval Test Failed: {e}")
    import traceback
    traceback.print_exc()
    print()
    sys.exit(1)

# ============================================================================
# TEST 7: End-to-End QA Test
# ============================================================================
print("TEST 7: End-to-End QA Chain Test")
print("-"*70)

try:
    from backend.rag.qa_chain import ValidatedQAChain
    from backend.vectorstore.pinecone_vector import PineconeManager
    from config.config import config
    
    # Initialize managers
    pinecone_mgr = PineconeManager(
        api_key=pinecone_key,
        index_name=index_name,
        embedding_model="text-embedding-3-large",
        dimension=3072
    )
    
    qa_chain = ValidatedQAChain(
        pinecone_manager=pinecone_mgr,
        similarity_threshold=0.3,  # Lower for testing
        confidence_threshold=0.4
    )
    
    # Test query
    test_question = "What is discussed in this document?"
    print(f"Question: '{test_question}'")
    print("Querying with threshold=0.3...\n")
    
    result = qa_chain.query(
        test_question,
        namespace=test_namespace
    )
    
    print(f"Answer: {result['answer'][:300]}...")
    print(f"\nConfidence: {result.get('confidence', 0):.0%}")
    print(f"Confidence Level: {result.get('confidence_level', 'N/A')}")
    print(f"Sources: {len(result.get('sources', []))}")
    print()
    
except Exception as e:
    print(f"❌ QA Chain Test Failed: {e}")
    import traceback
    traceback.print_exc()
    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print()
print("If all tests passed:")
print("  ✅ Your OpenAI key works")
print("  ✅ Your Pinecone connection works")
print("  ✅ Documents are indexed")
print("  ✅ Retrieval works")
print()
print("If retrieval found NO results:")
print("  1. Your documents may not contain 'gas clean agent' information")
print("  2. The similarity threshold in your app is too high")
print("  3. Try the suggestions below")
print()
print("RECOMMENDATIONS:")
print("  1. Set SIMILARITY_THRESHOLD=0.3 in .env")
print("  2. Check what's actually in your document")
print("  3. Use more specific terms from your actual PDF")
print()
print("="*70)