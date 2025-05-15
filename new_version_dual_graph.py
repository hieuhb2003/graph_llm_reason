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


# Cập nhật hàm llm_model_func để xử lý lỗi 429
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "google/gemini-2.0-flash-001",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="sk-or-v1-",
        base_url="https://openrouter.ai/api/v1",
        **kwargs
    )

async def embedding_func(texts, model):
    return model.encode(texts)["dense_vecs"]

# Chuyển lại sang hàm đồng bộ
def init_rag(working_dir, embedding_path, embedding_func_name, embedding_func_name_chunks ,devices):
    print("Initializing LightRAG...")
    print(f"Using LLM model: google/gemini-2.0-pro-exp-02-05:free")
    
    # Khởi tạo đối tượng LightRAG
    from FlagEmbedding import BGEM3FlagModel

    # model = BGEM3FlagModel('/home/hungpv/projects/train_embedding/nanographrag/flag_embedding_train/test_encoder_only_m3_bge-m3_sd/checkpoint-90804',  
    #                    use_fp16=False, devices = ["cuda:0"],pooling_method = "mean")
    model = BGEM3FlagModel(embedding_path,  
                       use_fp16=False, devices = devices,pooling_method = "mean")
    
    embedding_func_name = embedding_func_name
    # embedding_func_name = "bge-m3-only-entity-name-only-edge-description"
    model_chunks = BGEM3FlagModel("BAAI/bge-m3",  
                       use_fp16=False, devices = devices,pooling_method = "mean")
    

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=512,
            func=lambda texts: embedding_func(texts, model_chunks),
        ),
        embedding_func_name = embedding_func_name,
        embedding_func_name_chunks = embedding_func_name_chunks,
        entity_embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=512,
            func=lambda texts: embedding_func(texts, model),
        ),
        
    )


class DualGraphQuery:
    """
    Class for querying two knowledge graphs in different languages,
    mapping entities between them, and returning consolidated results.
    """
        
    def __init__(
        self,
        graph1_instance,
        graph2_instance,
        mapping_file_path: str,
        output_dir: str = "query_results",
        graph1_name: str = "graph1",
        graph2_name: str = "graph2",
        mapping_sym_file: Optional[str] = None
    ):
        """Initialize dual graph querying with two LightRAG instances and entity mapping."""
        self.graph1 = graph1_instance
        self.graph2 = graph2_instance
        self.mapping_file = mapping_file_path
        # self.output_dir = output_dir
        self.graph1_name = graph1_name
        self.graph2_name = graph2_name
        self.mapping_sym_file = mapping_sym_file
        
        # Create output directory if it doesn't exist
        # os.makedirs(output_dir, exist_ok=True)
        
        # Validate mapping file exists
        if not os.path.exists(mapping_file_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")

        # Initialize BatchGraphMapper for faster mapping
        self.batch_mapper = BatchGraphMapper(
            mapping_file_path=mapping_file_path,
            graph1_name=self.graph1_name,
            graph2_name=self.graph2_name
        )
        
        # Initialize BatchGraphMapper for symmetrical mapping if needed
        self.sym_mapper = None
        if mapping_sym_file and os.path.exists(mapping_sym_file):
            self.sym_mapper = BatchGraphMapper(
                mapping_file_path=mapping_sym_file,
                graph1_name=self.graph1_name,
                graph2_name=self.graph1_name
            )
        
        logger.info(f"DualGraphQuery initialized with {graph1_name} and {graph2_name}")
    
    def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True,
        sym: bool = False,
        max_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Query both graphs and return consolidated results.
        
        Args:
            query_text: The query string
            mode: Query mode - 'local' (nodes), 'global' (edges), or 'hybrid' (both)
            param: Query parameters (if None, will be created based on mode)
            save_results: Whether to save results to a JSONL file
            
        Returns:
            Dict containing consolidated query results
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery(query_text, mode, param, save_results, sym=sym, max_threshold=max_threshold)
        )
    
    async def aquery(
        self,
        query_text: str,
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True,
        sym: bool = False,
        index: Optional[int] = None,
        max_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Async version of query method.
        """
        # Set up query parameters based on mode if not provided
        if param is None:
            param = QueryParam()
            
            if mode.lower() == "local":
                param.mode = "local"
            elif mode.lower() == "global":
                param.mode = "global"
            else:
                param.mode = "hybrid"  # default
        
        logger.info(f"Querying with mode: {param.mode}")
        
        # Query both graphs
        graph2_results = await self._query_single_graph(self.graph2, query_text, param)
        # print("Graph2 results: ", graph2_results)
        # Map entities from graph2 to graph1
        mapped_graph2_results, _, _ = self._map_graph2_to_graph1(graph2_results)

        result = {"entities": {}, "edges": {}}
        # print(mapped_graph2_results)
        for item in mapped_graph2_results:
            # item = item[0]
            # print(item)
            if item["retrieval_type"] == "node":
                entity_id = item['hash_id']
                if entity_id not in result["entities"]:
                    result["entities"][entity_id] = {}
                    result["entities"][entity_id]['name'] = item["entity_name"]
                    result["entities"][entity_id]['description'] = item['description']
                    result["entities"][entity_id]['score'] = item['score']
            elif item["retrieval_type"] == "edge":
                edge_id = item['hash_id']
                if edge_id not in result["edges"]:
                    result["edges"][edge_id] = {}
                    result["edges"][edge_id]['src_id'] = item['src_id']
                    result["edges"][edge_id]['tgt_id'] = item['tgt_id']
                    result["edges"][edge_id]['description'] = item['description']
                    result["edges"][edge_id]['score'] = item['score']
        # print(result)   
        print("Len of entities: ", len(result["entities"]))
        print("Len of edges: ", len(result["edges"]))
        print("Len of graph2_results: ", len(mapped_graph2_results))      
        return result
    
    async def _query_single_graph(
        self, 
        graph_instance, 
        query_text: str, 
        param: QueryParam
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Query a single graph and return results."""
        try:
            candidates, hl_keywords, ll_keywords = await graph_instance.anew_retrieval(
                query=query_text,
                param=param
            )
            return candidates, hl_keywords, ll_keywords
        except Exception as e:
            logger.error(f"Error querying graph: {e}")
            return [], [], []
    

    def _map_graph2_to_graph1(
        self, 
        graph2_results: Tuple[List[Dict[str, Any]], List[str], List[str]]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Map entities and edges from graph2 to graph1 using the mapping file."""
        # Use the batch mapper for better performance
        return self.batch_mapper.map_graph2_to_graph1(graph2_results)
# Command-line execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query dual knowledge graphs with entity mapping")
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid"], 
                       help="Query mode - local (nodes), global (edges), or hybrid (both)")
    parser.add_argument("--graph1_path", required=True, help="Path to first graph's data directory")
    parser.add_argument("--graph2_path", required=True, help="Path to second graph's data directory")
    parser.add_argument("--mapping_file", required=True, help="Path to entity/edge mapping file")
    parser.add_argument("--mapping_sym_file", default=None, help="Path to entity/edge mapping file")
    parser.add_argument("--output_dir", default="query_results", help="Directory to save results")
    parser.add_argument("--graph1_name", default="graph1", help="Name for first graph")
    parser.add_argument("--graph2_name", default="graph2", help="Name for second graph")
    parser.add_argument("--batch_output", help="Filename for combined batch results (for batch mode)")
    parser.add_argument("--embedding_func_name", default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Embedding model to use for both graphs")
    parser.add_argument("--embedding_func_name_chunks", default="bge-m3-list_des", 
                        help="Embedding model to use for both graphs")
    parser.add_argument("--embedding_path", default="graph1", help="Name for first graph")
    parser.add_argument('--devices', nargs='+', help='List of CUDA devices')
    parser.add_argument("--threshold_sym", type=float, default=None)
    
    parser.add_argument('--corpus_file', type=str, default='/home/hungpv/projects/TN/data/data_nq/filter_corpus_en.json', help='Path to the corpus file')
    parser.add_argument('--queries_file', type=str, default='/home/hungpv/projects/TN/data/data_nq/dev_queries_vi.json', help='Path to the queries file')
    parser.add_argument('--output_file', type=str, default='/home/hungpv/projects/draft/nanographrag/bge_new_rerank_method/nq_q_vi/query_vi_result.json', help='Path to the output file')
    parser.add_argument('--need_edge', type=float, default=0, help='Path to the output file')
    parser.add_argument('--method', default=None, type=str) # max, max_top_k, new_function 
    parser.add_argument("--threshold", default=0.7, type=float)
    
    
    
    
    args = parser.parse_args()
    

# working_dir, embedding_path, embedding_func_name, embedding_func_name_chunks ,devices
    graph1 = init_rag(
        working_dir=args.graph1_path,
        embedding_path=args.embedding_path,
        embedding_func_name=args.embedding_func_name, 
        embedding_func_name_chunks=args.embedding_func_name_chunks,
        devices=args.devices,
    )
    graph2 = init_rag(
        working_dir=args.graph2_path,
        embedding_path=args.embedding_path,
        embedding_func_name=args.embedding_func_name, 
        embedding_func_name_chunks=args.embedding_func_name_chunks,
        devices=args.devices,
    )
    
    # Create dual query instance
    dual_query = DualGraphQuery(
        graph1_instance=graph1,
        graph2_instance=graph2,
        mapping_file_path=args.mapping_file,
        output_dir=args.output_dir,
        graph1_name=args.graph1_name,
        graph2_name=args.graph2_name,
        mapping_sym_file=args.mapping_sym_file
    )
    
    
    
    
    with open(args.corpus_file, 'r') as f:
        corpus = json.load(f)
    corpus_to_idx = {values: key for key, values in corpus.items()}
    
    chunks_dict = graph1.text_chunks._data
    docs_dict = graph1.full_docs._data

    # Read queries from file
    with open(args.queries_file, "r") as f:
        queries = json.load(f)

    # chosen_id = [f"query_{idx}" for idx in range(1)]
    # queries = {k: v for k, v in queries.items() if k in chosen_id}
    
    final_results = {}
    
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Run retrieval for each query
    for idx_query in queries:
        query = queries[idx_query]
        start_time = time.time()
        
        predefined_candidates = dual_query.query(
            query_text=query,
            mode=args.mode,
            save_results=True,
            sym=args.mapping_sym_file is not None,
            max_threshold=args.threshold_sym
        )
        
        results = graph1.retrieve_docs_with_enhanced_method(
            query=query,
            top_k=100,
            weight_chunk=1,
            weight_edge=args.need_edge,
            weight_entity=args.need_edge,
            method=args.method,
            threshold= args.threshold,
            predefined_candidates=predefined_candidates
        )

        final_results[query] = results

        # Save the results to file
        with open(args.output_file, "w") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

    
    
