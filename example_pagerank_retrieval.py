import os
import asyncio
import numpy as np
from datetime import datetime
import time
from lightrag.lightrag import LightRAG

# Simple example embedding function that returns random embeddings
# In production, you should use a real embedding model
async def example_embedding_func(texts):
    return np.random.rand(len(texts), 768)  # 768-dimensional embeddings

# Simple example LLM function
async def example_llm_model_func(prompt, hashing_kv=None, **kwargs):
    return f"Example response for: {prompt[:50]}..."

async def main():
    # Initialize LightRAG with our embedding and LLM functions
    rag = LightRAG(
        working_dir="./lightrag_cache",
        embedding_func=example_embedding_func,
        llm_model_func=example_llm_model_func,
    )
    
    # Phần 1: Kịch bản mặc định - tự động tính toán embeddings
    print("\n=== PHẦN 1: BẢN THƯỜNG - TỰ ĐỘNG TÍNH TOÁN EMBEDDINGS ===")
    
    # Precompute data for PageRank retrieval
    print("\n--- Step 1: Precomputing data for PageRank retrieval ---")
    start_time = time.time()
    await rag.precompute_pagerank_data()
    precompute_time = time.time() - start_time
    print(f"Precomputation completed in {precompute_time:.2f} seconds.")
    
    # Phần 2: Kịch bản chỉ dùng embeddings có sẵn
    print("\n=== PHẦN 2: CHỈ DÙNG EMBEDDINGS CÓ SẴN TRONG VECTOR DB ===")
    
    # Precompute với tùy chọn force_use_existing_embeddings
    print("\n--- Step 1: Precomputing data with force_use_existing_embeddings=True ---")
    start_time = time.time()
    await rag.precompute_pagerank_data(force_use_existing_embeddings=True)
    precompute_time_optimized = time.time() - start_time
    print(f"Optimized precomputation completed in {precompute_time_optimized:.2f} seconds.")
    
    # Basic usage example of PageRank retrieval
    test_query = "Tell me about machine learning"
    
    # Configure PageRank parameters 
    pagerank_config = {
        "linking_top_k_facts": 15,       # Number of top facts to consider
        "passage_node_weight_factor": 0.3, # Factor to weigh DPR scores for personalization
        "damping_factor": 0.5,          # Damping factor for PageRank
        "use_synonyms": False,          # Whether to use synonyms
        "direct_dpr_to_chunk_weight": 0.2, # Weight of direct DPR score in final chunk score
        "average_ppr_for_chunk": True,  # Whether to average PPR scores for chunks
    }
    
    # Truy vấn với tùy chọn force_use_existing_embeddings
    print("\n--- Step 2: Retrieve documents using precomputed data with force_use_existing_embeddings ---")
    start_time = time.time()
    results = await rag.retrieve_docs_with_pagerank(
        query=test_query,
        top_k=5,
        pagerank_config=pagerank_config,
        use_precomputed_data=True,
        force_use_existing_embeddings=True,
    )
    retrieval_time_optimized = time.time() - start_time
    print(f"Optimized retrieval completed in {retrieval_time_optimized:.2f} seconds.")
    
    # Display results
    print(f"\nResults for query: '{test_query}'")
    print("-" * 50)
    
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result['score']:.4f}):")
        print(f"Chunk ID: {result['id']}")
        # Print just a snippet of the content
        content_snippet = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        print(f"Content: {content_snippet}")
        print("-" * 50)
    
    # So sánh hiệu suất
    print("\n=== SO SÁNH HIỆU SUẤT ===")
    print(f"Precomputation time (regular): {precompute_time:.2f} seconds")
    print(f"Precomputation time (optimized): {precompute_time_optimized:.2f} seconds")
    if precompute_time > 0:
        print(f"Speedup: {precompute_time/precompute_time_optimized:.1f}x faster")
    
    print("\n--- TÌM HIỂU SỰ KHÁC BIỆT ---")
    print("Cách tiếp cận tối ưu chỉ sử dụng embeddings có sẵn trong vector database,")
    print("không tính toán lại cho các entities/facts/chunks không có trong DB.")
    print("Điều này giúp tiết kiệm đáng kể thời gian tính toán, nhưng có thể loại bỏ một số thông tin")
    print("mà chưa có embeddings trong vector DB.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 