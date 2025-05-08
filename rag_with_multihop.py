import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import asdict

# Import from lightrag components
from lightrag.utils import logger
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.operate_old import kg_direct_recall
from lightrag.base import QueryParam
from lightrag.lightrag_old import always_get_an_event_loop

# Import entity mapping
# from entity_mapping import find_mapped_entity_description, find_mapped_edge_description
# Import entity mapping
from optimized_entity_mapping import find_mapped_entity_description, find_mapped_edge_description, BatchGraphMapper


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
if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Query dual knowledge graphs with entity mapping")
    # parser.add_argument("--query", help="Query text (for single query mode)")
    # parser.add_argument("--queries_file", help="File containing one query per line (for batch mode)")
    # parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid"], 
    #                    help="Query mode - local (nodes), global (edges), or hybrid (both)")
    # parser.add_argument("--graph1_path", required=True, help="Path to first graph's data directory")
    # parser.add_argument("--graph2_path", required=True, help="Path to second graph's data directory")
    # parser.add_argument("--mapping_file", required=True, help="Path to entity/edge mapping file")
    # parser.add_argument("--mapping_sym_file", default=None, help="Path to entity/edge mapping file")
    # parser.add_argument("--output_dir", default="query_results", help="Directory to save results")
    # parser.add_argument("--graph1_name", default="graph1", help="Name for first graph")
    # parser.add_argument("--graph2_name", default="graph2", help="Name for second graph")
    # parser.add_argument("--batch_output", help="Filename for combined batch results (for batch mode)")
    # parser.add_argument("--embedding_func_name", default="sentence-transformers/all-MiniLM-L6-v2", 
    #                     help="Embedding model to use for both graphs")
    # parser.add_argument("--embedding_path", default="graph1", help="Name for first graph")
    # parser.add_argument('--devices', nargs='+', help='List of CUDA devices')
    # parser.add_argument("--threshold_sym", type=float, default=None)
    
    # args = parser.parse_args()
    
    # Ensure either query or queries_file is provided
    # if not args.query and not args.queries_file:
    #     parser.error("Either --query or --queries_file must be provided")
    

    graph = init_rag(
        working_dir="/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding",
        embedding_path="BAAI/bge-m3",
        embedding_func_name="bge-m3-list_des", 
        devices="cuda:0",
        # embedding_func=embedding_func
    )
    # query = "When was the region immediately north of the region where Israel is located and the location of the Battle of Qurah and Umm al Maradim created?"
    # result = graph.multi_hop_retrieval(query=query)
    # # print(result[:20])
    # r = set()
    # conut = 0 
    # for item in result:
    #     if item['retrieved_chunk_id'] not in r:
    #         print(item['retrieved_chunk_id'])
    #         r.add(item['retrieved_chunk_id'])
    #         conut +=1
    # print(conut)
    with open("/home/hungpv/projects/TN/data/data_nq/query_vi.json",'r') as f:
        queries = json.load(f)
    from tqdm import tqdm
    for i,query in tqdm(enumerate(queries)):
        print(f"Query {i+1}: {query}")
        result = graph.multi_hop_retrieval(query=query)
        # print(result[:20])
        item = {
            'query': query,
            "candidates": result
        }
        with open(f"/home/hungpv/projects/draft/nanographrag/new_nq_llm_vi/query_{i+1}.jsonl", 'w', encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False))
                