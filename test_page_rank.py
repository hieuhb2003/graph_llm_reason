import os
import sys
import time
import json
import random
import asyncio
from typing import List


from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import detect_language

from lightrag.utils import logger
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.operate_old import kg_direct_recall
from lightrag.base import QueryParam
from lightrag.lightrag_old import always_get_an_event_loop
# Simple example embedding function that returns random embeddings
# In production, you should use a real embedding model
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "google/gemini-2.0-flash-001",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="",
        base_url="https://openrouter.ai/api/v1",
        **kwargs
    )

async def embedding_func(texts, model):
    return model.encode(texts)["dense_vecs"]

# Chuyển lại sang hàm đồng bộ
def init_rag(working_dir, embedding_path, embedding_func_name, devices):
    print("Initializing LightRAG...")
    print(f"google/gemini-2.0-flash-lite-001")
    
    # Khởi tạo đối tượng LightRAG
    from FlagEmbedding import BGEM3FlagModel

    # model = BGEM3FlagModel('/home/hungpv/projects/train_embedding/nanographrag/flag_embedding_train/test_encoder_only_m3_bge-m3_sd/checkpoint-90804',  
    #                    use_fp16=False, devices = ["cuda:0"],pooling_method = "mean")
    model = BGEM3FlagModel(embedding_path,  
                       use_fp16=False, devices = devices,pooling_method = "mean")
    
    embedding_func_name = embedding_func_name
    # embedding_func_name = "bge-m3-only-entity-name-only-edge-description"
# /home/hungpv/projects/TN/LIGHTRAG/zalo_wiki_graph_single_vi_v2/vdb_chunks_fine-tune-embedding.json

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=512,
            func=lambda texts: embedding_func(texts, model),
        ),
        embedding_func_name = embedding_func_name
    )

# async def main():
#     # Initialize LightRAG with our embedding and LLM functions
#     rag = LightRAG(
#         working_dir="./lightrag_cache",
#         embedding_func=example_embedding_func,
#         llm_model_func=example_llm_model_func,
#     )
    
#     # Phần 1: Kịch bản mặc định - tự động tính toán embeddings
#     print("\n=== PHẦN 1: BẢN THƯỜNG - TỰ ĐỘNG TÍNH TOÁN EMBEDDINGS ===")
    
#     # Precompute data for PageRank retrieval
#     print("\n--- Step 1: Precomputing data for PageRank retrieval ---")
#     start_time = time.time()
#     await rag.precompute_pagerank_data()
#     precompute_time = time.time() - start_time
#     print(f"Precomputation completed in {precompute_time:.2f} seconds.")
    
#     # Phần 2: Kịch bản chỉ dùng embeddings có sẵn
#     print("\n=== PHẦN 2: CHỈ DÙNG EMBEDDINGS CÓ SẴN TRONG VECTOR DB ===")
    
#     # Precompute với tùy chọn force_use_existing_embeddings
#     print("\n--- Step 1: Precomputing data with force_use_existing_embeddings=True ---")
#     start_time = time.time()
#     await rag.precompute_pagerank_data(force_use_existing_embeddings=True)
#     precompute_time_optimized = time.time() - start_time
#     print(f"Optimized precomputation completed in {precompute_time_optimized:.2f} seconds.")
    
#     # Basic usage example of PageRank retrieval
#     test_query = "Tell me about machine learning"
    
#     # Configure PageRank parameters 
#     pagerank_config = {
#         "linking_top_k_facts": 15,       # Number of top facts to consider
#         "passage_node_weight_factor": 0.3, # Factor to weigh DPR scores for personalization
#         "damping_factor": 0.5,          # Damping factor for PageRank
#         "use_synonyms": False,          # Whether to use synonyms
#         "direct_dpr_to_chunk_weight": 0.2, # Weight of direct DPR score in final chunk score
#         "average_ppr_for_chunk": True,  # Whether to average PPR scores for chunks
#     }
    
#     # Truy vấn với tùy chọn force_use_existing_embeddings
#     print("\n--- Step 2: Retrieve documents using precomputed data with force_use_existing_embeddings ---")
#     start_time = time.time()
#     results = await rag.retrieve_docs_with_pagerank(
#         query=test_query,
#         top_k=5,
#         pagerank_config=pagerank_config,
#         use_precomputed_data=True,
#         force_use_existing_embeddings=True,
#     )
#     retrieval_time_optimized = time.time() - start_time
#     print(f"Optimized retrieval completed in {retrieval_time_optimized:.2f} seconds.")
    
#     # Display results
#     print(f"\nResults for query: '{test_query}'")
#     print("-" * 50)
    
#     for i, result in enumerate(results):
#         print(f"Result {i+1} (Score: {result['score']:.4f}):")
#         print(f"Chunk ID: {result['id']}")
#         # Print just a snippet of the content
#         content_snippet = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
#         print(f"Content: {content_snippet}")
#         print("-" * 50)
    
#     # So sánh hiệu suất
#     print("\n=== SO SÁNH HIỆU SUẤT ===")
#     print(f"Precomputation time (regular): {precompute_time:.2f} seconds")
#     print(f"Precomputation time (optimized): {precompute_time_optimized:.2f} seconds")
#     if precompute_time > 0:
#         print(f"Speedup: {precompute_time/precompute_time_optimized:.1f}x faster")
    
#     print("\n--- TÌM HIỂU SỰ KHÁC BIỆT ---")
#     print("Cách tiếp cận tối ưu chỉ sử dụng embeddings có sẵn trong vector database,")
#     print("không tính toán lại cho các entities/facts/chunks không có trong DB.")
#     print("Điều này giúp tiết kiệm đáng kể thời gian tính toán, nhưng có thể loại bỏ một số thông tin")
#     print("mà chưa có embeddings trong vector DB.")

if __name__ == "__main__":
    # Run the async main function
    graph_rag = init_rag(working_dir= "/home/hungpv/projects/TN/LIGHTRAG/new_musique_en_without_embedding",
                        embedding_path="BAAI/bge-m3", 
                        embedding_func_name='bge-m3-list_des', 
                        devices='cuda:0')
    # retrieve_docs_with_pagerank
    test_query = "Khi nào người mà những bàn thắng của Messi ở Copa del Rey được so sánh để ký hợp đồng với Barcelona?"
    
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
    results =  graph_rag.retrieve_docs_with_pagerank(
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
    print("Thời gian: ", retrieval_time_optimized)