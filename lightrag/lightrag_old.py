from __future__ import annotations
import json
import asyncio
import os
import configparser
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Any, AsyncIterator, Callable, Iterator, cast, List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity  
import numpy as np        
import re
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
)
import networkx as nx
from collections import defaultdict
from .namespace_old import NameSpace, make_namespace
from .operate_old import (
    chunking_by_token_size,
    extract_entities,
    extract_keywords_only,
    kg_query,
    kg_query_with_keywords,
    mix_kg_vector_query,
    naive_query,
    kg_retrieval,
    _merge_nodes_then_upsert,
    _merge_edges_then_upsert,
    naive_retrieval,
    kg_direct_recall,
    _most_relevant_text_chunks_from_nodes,
    _most_relevant_text_chunks_from_edges, # New function returning (candidates, hl_keywords, ll_keywords)
    # enhanced_chunk_retrieval,
    enhanced_chunk_retrieval_direct
)


from .prompt_old import GRAPH_FIELD_SEP, PROMPTS
from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    convert_response_to_json,
    limit_async_func_call,
    logger,
    set_logger,
    detect_language,
)
from .types import KnowledgeGraph

from tqdm import tqdm
from typing import Optional, List

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")
import time

# Storage type and implementation compatibility validation table
STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": [
            "JsonKVStorage",
            "MongoKVStorage",
            "RedisKVStorage",
            "TiDBKVStorage",
            "PGKVStorage",
            "OracleKVStorage",
        ],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": [
            "NetworkXStorage",
            "Neo4JStorage",
            "MongoGraphStorage",
            "TiDBGraphStorage",
            "AGEStorage",
            "GremlinStorage",
            "PGGraphStorage",
            "OracleGraphStorage",
        ],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    "VECTOR_STORAGE": {
        "implementations": [
            "NanoVectorDBStorage",
            "MilvusVectorDBStorage",
            "ChromaVectorDBStorage",
            "TiDBVectorDBStorage",
            "PGVectorStorage",
            "FaissVectorDBStorage",
            "QdrantVectorDBStorage",
            "OracleVectorDBStorage",
            "MongoVectorDBStorage",
        ],
        "required_methods": ["query", "upsert"],
    },
    "DOC_STATUS_STORAGE": {
        "implementations": [
            "JsonDocStatusStorage",
            "PGDocStatusStorage",
            "PGDocStatusStorage",
            "MongoDocStatusStorage",
        ],
        "required_methods": ["get_docs_by_status"],
    },
}

# Storage implementation environment variable without default value
STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    # KV Storage Implementations
    "JsonKVStorage": [],
    "MongoKVStorage": [],
    "RedisKVStorage": ["REDIS_URI"],
    "TiDBKVStorage": ["TIDB_USER", "TIDB_PASSWORD", "TIDB_DATABASE"],
    "PGKVStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "OracleKVStorage": [
        "ORACLE_DSN",
        "ORACLE_USER",
        "ORACLE_PASSWORD",
        "ORACLE_CONFIG_DIR",
    ],
    # Graph Storage Implementations
    "NetworkXStorage": [],
    "Neo4JStorage": ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"],
    "MongoGraphStorage": [],
    "TiDBGraphStorage": ["TIDB_USER", "TIDB_PASSWORD", "TIDB_DATABASE"],
    "AGEStorage": [
        "AGE_POSTGRES_DB",
        "AGE_POSTGRES_USER",
        "AGE_POSTGRES_PASSWORD",
    ],
    "GremlinStorage": ["GREMLIN_HOST", "GREMLIN_PORT", "GREMLIN_GRAPH"],
    "PGGraphStorage": [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    ],
    "OracleGraphStorage": [
        "ORACLE_DSN",
        "ORACLE_USER",
        "ORACLE_PASSWORD",
        "ORACLE_CONFIG_DIR",
    ],
    # Vector Storage Implementations
    "NanoVectorDBStorage": [],
    "MilvusVectorDBStorage": [],
    "ChromaVectorDBStorage": [],
    "TiDBVectorDBStorage": ["TIDB_USER", "TIDB_PASSWORD", "TIDB_DATABASE"],
    "PGVectorStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "FaissVectorDBStorage": [],
    "QdrantVectorDBStorage": ["QDRANT_URL"],  # QDRANT_API_KEY has default value None
    "OracleVectorDBStorage": [
        "ORACLE_DSN",
        "ORACLE_USER",
        "ORACLE_PASSWORD",
        "ORACLE_CONFIG_DIR",
    ],
    "MongoVectorDBStorage": [],
    # Document Status Storage Implementations
    "JsonDocStatusStorage": [],
    "PGDocStatusStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    "MongoDocStatusStorage": [],
}

# Storage implementation module mapping
STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "JsonDocStatusStorage": ".kg.json_doc_status_impl",
    "Neo4JStorage": ".kg.neo4j_impl",
    "OracleKVStorage": ".kg.oracle_impl",
    "OracleGraphStorage": ".kg.oracle_impl",
    "OracleVectorDBStorage": ".kg.oracle_impl",
    "MilvusVectorDBStorage": ".kg.milvus_impl",
    "MongoKVStorage": ".kg.mongo_impl",
    "MongoDocStatusStorage": ".kg.mongo_impl",
    "MongoGraphStorage": ".kg.mongo_impl",
    "MongoVectorDBStorage": ".kg.mongo_impl",
    "RedisKVStorage": ".kg.redis_impl",
    "ChromaVectorDBStorage": ".kg.chroma_impl",
    "TiDBKVStorage": ".kg.tidb_impl",
    "TiDBVectorDBStorage": ".kg.tidb_impl",
    "TiDBGraphStorage": ".kg.tidb_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "AGEStorage": ".kg.age_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "GremlinStorage": ".kg.gremlin_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
    "FaissVectorDBStorage": ".kg.faiss_impl",
    "QdrantVectorDBStorage": ".kg.qdrant_impl",
}


def lazy_external_import(module_name: str, class_name: str) -> Callable[..., Any]:
    """Lazily import a class from an external module based on the package of the caller."""
    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args: Any, **kwargs: Any):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop

def check_exist_embedding_func(embedding_func_name, working_dir):
    all_file_in_working_dirs = os.listdir(working_dir)
    entities_vdb_path = "vdb_" + NameSpace.VECTOR_STORE_ENTITIES + "_" + embedding_func_name + ".json"
    relationships_vdb_path = "vdb_" + NameSpace.VECTOR_STORE_RELATIONSHIPS+ "_" + embedding_func_name + ".json"
    chunks_vdb_path = "vdb_" + NameSpace.VECTOR_STORE_CHUNKS + "_" + embedding_func_name + ".json"

    if entities_vdb_path in all_file_in_working_dirs \
            and relationships_vdb_path in all_file_in_working_dirs \
            and chunks_vdb_path in all_file_in_working_dirs:

        return True
    return False


@dataclass
class LightRAG:
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    working_dir: str = field(
        default_factory=lambda: f"./lightrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    """Directory where cache and temporary files are stored."""

    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache.
    - enabled: If True, enables caching to avoid redundant computations.
    - similarity_threshold: Minimum similarity score to use cached embeddings.
    - use_llm_check: If True, validates cached embeddings using an LLM.

    """

    cache_queries_file : str = field(default = None)
    cache_queries: dict = field(default = None)
    embedding_func_name : str = field(default = None)

    kv_storage: str = field(default="JsonKVStorage")
    """Storage backend for key-value data."""

    vector_storage: str = field(default="NanoVectorDBStorage")
    """Storage backend for vector embeddings."""

    graph_storage: str = field(default="NetworkXStorage")
    """Storage backend for knowledge graphs."""

    doc_status_storage: str = field(default="JsonDocStatusStorage")
    """Storage type for tracking document processing statuses."""

    # Logging
    current_log_level = logger.level
    log_level: int = field(default=current_log_level)
    """Logging level for the system (e.g., 'DEBUG', 'INFO', 'WARNING')."""

    log_dir: str = field(default=os.getcwd())
    """Directory where logs are stored. Defaults to the current working directory."""

    # Text chunking
    chunk_token_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    """Maximum number of tokens per text chunk when splitting documents."""

    chunk_overlap_token_size: int = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))
    """Number of overlapping tokens between consecutive text chunks to preserve context."""

    tiktoken_model_name: str = "gpt-4o-mini"
    """Model name used for tokenization when chunking text."""

    # Entity extraction
    entity_extract_max_gleaning: int = 1
    """Maximum number of entity extraction attempts for ambiguous content."""

    entity_summary_to_max_tokens: int = int(os.getenv("MAX_TOKEN_SUMMARY", "500"))
    """Maximum number of tokens used for summarizing extracted entities."""

    # Node embedding
    node_embedding_algorithm: str = "node2vec"
    """Algorithm used for node embedding in knowledge graphs."""

    node2vec_params: dict[str, int] = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    """Configuration for the node2vec embedding algorithm:
    - dimensions: Number of dimensions for embeddings.
    - num_walks: Number of random walks per node.
    - walk_length: Number of steps per random walk.
    - window_size: Context window size for training.
    - iterations: Number of iterations for training.
    - random_seed: Seed value for reproducibility.
    """

    embedding_func: EmbeddingFunc | None = None
    """Function for computing text embeddings. Must be set before use."""

    embedding_batch_num: int = 32
    """Batch size for embedding computations."""

    embedding_func_max_async: int = 16
    """Maximum number of concurrent embedding function calls."""

    # LLM Configuration
    llm_model_func: Callable[..., object] | None = None
    """Function for interacting with the large language model (LLM). Must be set before use."""

    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    """Name of the LLM model used for generating responses."""

    llm_model_max_token_size: int = int(os.getenv("MAX_TOKENS", "32768"))
    """Maximum number of tokens allowed per LLM response."""

    llm_model_max_async: int = int(os.getenv("MAX_ASYNC", "16"))
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    # Storage
    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    namespace_prefix: str = field(default="")
    """Prefix for namespacing stored data across different environments."""

    enable_llm_cache: bool = True
    """Enables caching for LLM responses to avoid redundant computations."""

    enable_llm_cache_for_entity_extract: bool = True
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    # Extensions
    addon_params: dict[str, Any] = field(default_factory=dict)

    # Storages Management
    auto_manage_storages_states: bool = True
    """If True, lightrag will automatically calls initialize_storages and finalize_storages at the appropriate times."""

    """Dictionary for additional parameters and extensions."""
    convert_response_to_json_func: Callable[[str], dict[str, Any]] = (
        convert_response_to_json
    )

    # Custom Chunking Function
    chunking_func: Callable[
        [
            str,
            str | None,
            bool,
            int,
            int,
            str,
        ],
        list[dict[str, Any]],
    ] = chunking_by_token_size

    def verify_storage_implementation(
        self, storage_type: str, storage_name: str
    ) -> None:
        """Verify if storage implementation is compatible with specified storage type

        Args:
            storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
            storage_name: Storage implementation name

        Raises:
            ValueError: If storage implementation is incompatible or missing required methods
        """
        if storage_type not in STORAGE_IMPLEMENTATIONS:
            raise ValueError(f"Unknown storage type: {storage_type}")

        storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
        if storage_name not in storage_info["implementations"]:
            raise ValueError(
                f"Storage implementation '{storage_name}' is not compatible with {storage_type}. "
                f"Compatible implementations are: {', '.join(storage_info['implementations'])}"
            )

    def check_storage_env_vars(self, storage_name: str) -> None:
        """Check if all required environment variables for storage implementation exist

        Args:
            storage_name: Storage implementation name

        Raises:
            ValueError: If required environment variables are missing
        """
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            raise ValueError(
                f"Storage implementation '{storage_name}' requires the following "
                f"environment variables: {', '.join(missing_vars)}"
            )

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, "lightrag.log")
        set_logger(log_file)

        # specify the cache queries
        if self.cache_queries_file is None:
            self.cache_queries_file = os.path.join(self.working_dir, "cache_queries.json")
        
        import json
        self.cache_queries =  {}
        with open(self.cache_queries_file, "r", encoding="utf-8") as f:
            for line in f:
                line_dict = json.loads(line)
                self.cache_queries.update(line_dict)


        if not os.path.exists(self.cache_queries_file):
            with open(self.cache_queries_file, 'w', encoding='utf-8'):
                pass

        

        logger.setLevel(self.log_level)
        logger.info(f"Logger initialized for working directory: {self.working_dir}")
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            self.verify_storage_implementation(storage_type, storage_name)
            # Check environment variables
            # self.check_storage_env_vars(storage_name)

        # Ensure vector_db_storage_cls_kwargs has required fields
        default_vector_db_kwargs = {
            "cosine_better_than_threshold": float(os.getenv("COSINE_THRESHOLD", "0.05"))
        }
        self.vector_db_storage_cls_kwargs = {
            **default_vector_db_kwargs,
            **self.vector_db_storage_cls_kwargs,
        }

        # Life cycle
        self.storages_status = StoragesStatus.NOT_CREATED

        # Show config
        global_config = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in global_config.items()])
        logger.debug(f"LightRAG init with param:\n  {_print_config}\n")

        # Init LLM
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(  # type: ignore
            self.embedding_func
        )

        # Initialize all storages
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = (
            self._get_storage_class(self.kv_storage)
        )  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(
            self.vector_storage
        )  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(
            self.graph_storage
        )  # type: ignore
        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls, global_config=global_config
        )
        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls, global_config=global_config
        )
        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls, global_config=global_config
        )

        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

        self.llm_response_cache: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
            ),
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_FULL_DOCS
            ),
            embedding_func=self.embedding_func,
        )
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.KV_STORE_TEXT_CHUNKS
            ),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
            ),
            embedding_func=self.embedding_func,
        )

        if self.embedding_func_name and check_exist_embedding_func(self.embedding_func_name, self.working_dir):
            
            self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES + "_" + self.embedding_func_name
            ),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name"},
        )
            self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS+ "_" + self.embedding_func_name
                ),
                embedding_func=self.embedding_func,
                meta_fields={"src_id", "tgt_id"},
            )
            self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS+ "_" + self.embedding_func_name
                ),
                embedding_func=self.embedding_func,
            )

        else:

            self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES
                ),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS
                ),
                embedding_func=self.embedding_func,
                meta_fields={"src_id", "tgt_id"},
            )
            self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS
                ),
                embedding_func=self.embedding_func,
            )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=make_namespace(self.namespace_prefix, NameSpace.DOC_STATUS),
            global_config=global_config,
            embedding_func=None,
        )

        if self.llm_response_cache and hasattr(
            self.llm_response_cache, "global_config"
        ):
            hashing_kv = self.llm_response_cache
        else:
            hashing_kv = self.key_string_value_json_storage_cls(  # type: ignore
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,  # type: ignore
                hashing_kv=hashing_kv,
                **self.llm_model_kwargs,
            )
        )

        self.storages_status = StoragesStatus.CREATED

        # Initialize storages
        if self.auto_manage_storages_states:
            loop = always_get_an_event_loop()
            loop.run_until_complete(self.initialize_storages())

    def __del__(self):
        # Finalize storages
        # if self.auto_manage_storages_states:
        #     print("Debug info:")
        #     print(f"llm_model_func is None: {self.llm_model_func is None}")
        #     print(f"embedding_func is None: {self.embedding_func is None}")
        #     loop = always_get_an_event_loop()
        #     loop.run_until_complete(self.finalize_storages())
        pass

    async def initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self.storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self.storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    async def finalize_storages(self):
        """Asynchronously finalize the storages"""
        if self.storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.finalize())

            await asyncio.gather(*tasks)

            self.storages_status = StoragesStatus.FINALIZED
            logger.debug("Finalized Storages")

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self, nodel_label: str, max_depth: int
    ) -> KnowledgeGraph:
        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label=nodel_label, max_depth=max_depth
        )

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        language: str = "Vietnamese" ,
        matching_method: str = "hybrid",
        need_cross_language: bool = True,
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """
        self.addon_params["current_language"] = language
        self.addon_params["matching_method"] = matching_method
        self.addon_params["need_cross_language"] = need_cross_language
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(input, split_by_character, split_by_character_only, language, matching_method)
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        language: str = "Vietnamese" , # Add language parameter
        matching_method: str = "hybrid"
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
        """
        self.addon_params["current_language"] = language
        self.addon_params["matching_method"] = matching_method
        
        await self.apipeline_enqueue_documents(input)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

    def insert_custom_chunks(self, full_text: str, text_chunks: list[str]) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_custom_chunks(full_text, text_chunks))

    async def ainsert_custom_chunks(
        self, full_text: str, text_chunks: list[str]
    ) -> None:
        update_storage = False
        try:
            doc_key = compute_mdhash_id(full_text.strip(), prefix="doc-")
            new_docs = {doc_key: {"content": full_text.strip()}}

            _add_doc_keys = await self.full_docs.filter_keys(set(doc_key))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("This document is already in the storage.")
                return

            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks: dict[str, Any] = {}
            for chunk_text in text_chunks:
                chunk_text_stripped = chunk_text.strip()
                chunk_key = compute_mdhash_id(chunk_text_stripped, prefix="chunk-")

                inserting_chunks[chunk_key] = {
                    "content": chunk_text_stripped,
                    "full_doc_id": doc_key,
                }

            doc_ids = set(inserting_chunks.keys())
            add_chunk_keys = await self.text_chunks.filter_keys(doc_ids)
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage.")
                return

            tasks = [
                self.chunks_vdb.upsert(inserting_chunks),
                self._process_entity_relation_graph(inserting_chunks),
                self.full_docs.upsert(new_docs),
                self.text_chunks.upsert(inserting_chunks),
            ]
            await asyncio.gather(*tasks)

        finally:
            if update_storage:
                await self._insert_done()

    # async def apipeline_enqueue_documents(self, input: str | list[str]) -> None:
    #     """
    #     Pipeline for Processing Documents

    #     1. Remove duplicate contents from the list
    #     2. Generate document IDs and initial status
    #     3. Filter out already processed documents
    #     4. Enqueue document in status
    #     """
    #     if isinstance(input, str):
    #         input = [input]

    #     # 1. Remove duplicate contents from the list
    #     unique_contents = list(set(doc.strip() for doc in input))

    #     # 2. Generate document IDs and initial status
    #     new_docs: dict[str, Any] = {
    #         compute_mdhash_id(content, prefix="doc-"): {
    #             "content": content,
    #             "content_summary": self._get_content_summary(content),
    #             "content_length": len(content),
    #             "status": DocStatus.PENDING,
    #             "created_at": datetime.now().isoformat(),
    #             "updated_at": datetime.now().isoformat(),
    #         }
    #         for content in unique_contents
    #     }

    #     # 3. Filter out already processed documents
    #     # Get docs ids
    #     all_new_doc_ids = set(new_docs.keys())
    #     # Exclude IDs of documents that are already in progress
    #     unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)
    #     # Filter new_docs to only include documents with unique IDs
    #     new_docs = {doc_id: new_docs[doc_id] for doc_id in unique_new_doc_ids}

    #     if not new_docs:
    #         logger.info("No new unique documents were found.")
    #         return

    #     # 4. Store status document
    #     await self.doc_status.upsert(new_docs)
    #     logger.info(f"Stored {len(new_docs)} new unique documents")

    # async def apipeline_process_enqueue_documents(
    #     self,
    #     split_by_character: str | None = None,
    #     split_by_character_only: bool = False,
    # ) -> None:
    #     """
    #     Process pending documents by splitting them into chunks, processing
    #     each chunk for entity and relation extraction, and updating the
    #     document status.

    #     1. Get all pending, failed, and abnormally terminated processing documents.
    #     2. Split document content into chunks
    #     3. Process each chunk for entity and relation extraction
    #     4. Update the document status
    #     """
    #     # 1. Get all pending, failed, and abnormally terminated processing documents.
    #     to_process_docs: dict[str, DocProcessingStatus] = {}

    #     processing_docs = await self.doc_status.get_docs_by_status(DocStatus.PROCESSING)
    #     to_process_docs.update(processing_docs)
    #     failed_docs = await self.doc_status.get_docs_by_status(DocStatus.FAILED)
    #     to_process_docs.update(failed_docs)
    #     pendings_docs = await self.doc_status.get_docs_by_status(DocStatus.PENDING)
    #     to_process_docs.update(pendings_docs)

    #     if not to_process_docs:
    #         logger.info("All documents have been processed or are duplicates")
    #         return

    #     # 2. split docs into chunks, insert chunks, update doc status
    #     batch_size = self.addon_params.get("insert_batch_size", 10)
    #     docs_batches = [
    #         list(to_process_docs.items())[i : i + batch_size]
    #         for i in range(0, len(to_process_docs), batch_size)
    #     ]

    #     logger.info(f"Number of batches to process: {len(docs_batches)}.")

    #     # 3. iterate over batches
    #     from tqdm import tqdm

    #     for batch_idx, docs_batch in tqdm(enumerate(docs_batches), desc="Processing batches", total=len(docs_batches)):
  
    #         # 4. iterate over batch
    #         for doc_id_processing_status in docs_batch:
    #             doc_id, status_doc = doc_id_processing_status
    #             # Update status in processing
    #             doc_status_id = compute_mdhash_id(status_doc.content, prefix="doc-")
    #             await self.doc_status.upsert(
    #                 {
    #                     doc_status_id: {
    #                         "status": DocStatus.PROCESSING,
    #                         "updated_at": datetime.now().isoformat(),
    #                         "content": status_doc.content,
    #                         "content_summary": status_doc.content_summary,
    #                         "content_length": status_doc.content_length,
    #                         "created_at": status_doc.created_at,
    #                     }
    #                 }
    #             )
    #             # Generate chunks from document
    #             chunks: dict[str, Any] = {
    #                 compute_mdhash_id(dp["content"], prefix="chunk-"): {
    #                     **dp,
    #                     "full_doc_id": doc_id,
    #                 }
    #                 for dp in self.chunking_func(
    #                     status_doc.content,
    #                     split_by_character,
    #                     split_by_character_only,
    #                     self.chunk_overlap_token_size,
    #                     self.chunk_token_size,
    #                     self.tiktoken_model_name,
    #                 )
    #             }

    #             # Process document (text chunks and full docs) in parallel
    #             tasks = [
    #                 self.chunks_vdb.upsert(chunks),
    #                 self._process_entity_relation_graph(chunks),
    #                 self.full_docs.upsert({doc_id: {"content": status_doc.content}}),
    #                 self.text_chunks.upsert(chunks),
    #             ]
    #             try:
    #                 await asyncio.gather(*tasks)
    #                 await self.doc_status.upsert(
    #                     {
    #                         doc_status_id: {
    #                             "status": DocStatus.PROCESSED,
    #                             "chunks_count": len(chunks),
    #                             "content": status_doc.content,
    #                             "content_summary": status_doc.content_summary,
    #                             "content_length": status_doc.content_length,
    #                             "created_at": status_doc.created_at,
    #                             "updated_at": datetime.now().isoformat(),
    #                         }
    #                     }
    #                 )
    #                 await self._insert_done()

    #             except Exception as e:
    #                 logger.error(f"Failed to process document {doc_id}: {str(e)}")
    #                 await self.doc_status.upsert(
    #                     {
    #                         doc_status_id: {
    #                             "status": DocStatus.FAILED,
    #                             "error": str(e),
    #                             "content": status_doc.content,
    #                             "content_summary": status_doc.content_summary,
    #                             "content_length": status_doc.content_length,
    #                             "created_at": status_doc.created_at,
    #                             "updated_at": datetime.now().isoformat(),
    #                         }
    #                     }
    #                 )
    #                 continue
    #         logger.info(f"Completed batch {batch_idx + 1} of {len(docs_batches)}.")

    async def apipeline_enqueue_documents(self, input: str | list[str]) -> None:
        """
        Pipeline for Processing Documents

        1. Remove duplicate contents from the list
        2. Generate document IDs and initial status
        3. Filter out already processed documents
        4. Enqueue document in status
        """
        if isinstance(input, str):
            input = [input]

        # 1. Remove duplicate contents from the list
        unique_contents = list(set(doc.strip() for doc in input))

        # 2. Generate document IDs and initial status
        new_docs: dict[str, Any] = {
            compute_mdhash_id(content, prefix="doc-"): {
                "content": content,
                "content_summary": self._get_content_summary(content),
                "content_length": len(content),
                "status": DocStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            for content in unique_contents
        }

        # 3. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already in progress
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)
        # Filter new_docs to only include documents with unique IDs
        new_docs = {doc_id: new_docs[doc_id] for doc_id in unique_new_doc_ids}

        if not new_docs:
            logger.info("No new unique documents were found.")
            return

        # 4. Store status document
        await self.doc_status.upsert(new_docs)
        logger.info(f"Stored {len(new_docs)} new unique documents")

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks and processing
        all chunks in parallel for entity and relation extraction.

        1. Get all pending, failed, and abnormally terminated processing documents
        2. Update all documents to processing status
        3. Split all documents into chunks
        4. Process all chunks in parallel
        5. Update all document statuses
        """
        # 1. Get all pending, failed, and abnormally terminated processing documents
        to_process_docs_full: dict[str, DocProcessingStatus] = {}

        processing_docs = await self.doc_status.get_docs_by_status(DocStatus.PROCESSING)
        to_process_docs_full.update(processing_docs)
        failed_docs = await self.doc_status.get_docs_by_status(DocStatus.FAILED)
        to_process_docs_full.update(failed_docs)
        pendings_docs = await self.doc_status.get_docs_by_status(DocStatus.PENDING)
        to_process_docs_full.update(pendings_docs)

        if not to_process_docs_full:
            logger.info("All documents have been processed or are duplicates")
            return

        batch_size = self.addon_params.get("insert_batch_size", 10)
        docs_batches = [
            list(to_process_docs_full.items())[i : i + batch_size]
            for i in range(0, len(to_process_docs_full), batch_size)
        ]

        for to_process_docs in docs_batches:
            logger.info(f"Number of batches to process: {len(docs_batches)}.")

            # 2. Update all documents to processing status
            status_updates = {}
            for doc_id, status_doc in to_process_docs:
                doc_status_id = compute_mdhash_id(status_doc.content, prefix="doc-")
                status_updates[doc_status_id] = {
                    "status": DocStatus.PROCESSING,
                    "updated_at": datetime.now().isoformat(),
                    "content": status_doc.content,
                    "content_summary": status_doc.content_summary,
                    "content_length": status_doc.content_length,
                    "created_at": status_doc.created_at,
                }
            
            await self.doc_status.upsert(status_updates)
            logger.info(f"Updated {len(status_updates)} documents to processing status")

            # 3. Generate all chunks from all documents
            all_chunks = {}
            doc_chunk_mapping = {}  # Maps doc_id to its chunks for later status updates
            
            for doc_id, status_doc in to_process_docs:
                doc_chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_id,
                    }
                    for dp in self.chunking_func(
                        status_doc.content,
                        split_by_character,
                        split_by_character_only,
                        self.chunk_overlap_token_size,
                        self.chunk_token_size,
                        self.tiktoken_model_name,
                    )
                }
                
                all_chunks.update(doc_chunks)
                doc_chunk_mapping[doc_id] = {
                    "status_doc": status_doc,
                    "chunk_count": len(doc_chunks)
                }
            
            logger.info(f"Generated {len(all_chunks)} chunks from {len(to_process_docs)} documents")

            # 4. Process all chunks in parallel
            try:
                # Upload all chunks to vector database
                await self.chunks_vdb.upsert(all_chunks)
                
                # Process all entity-relation graphs in parallel
                await self._process_entity_relation_graph(all_chunks)
                
                # Upload all full documents
                full_docs_upload = {
                    doc_id: {"content": status_doc.content} 
                    for doc_id, status_doc in to_process_docs
                }
                await self.full_docs.upsert(full_docs_upload)
                
                # Upload all text chunks
                await self.text_chunks.upsert(all_chunks)
                
                # 5. Update all document statuses to processed
                success_updates = {}
                for doc_id, mapping in doc_chunk_mapping.items():
                    status_doc = mapping["status_doc"]
                    doc_status_id = compute_mdhash_id(status_doc.content, prefix="doc-")
                    success_updates[doc_status_id] = {
                        "status": DocStatus.PROCESSED,
                        "chunks_count": mapping["chunk_count"],
                        "content": status_doc.content,
                        "content_summary": status_doc.content_summary,
                        "content_length": status_doc.content_length,
                        "created_at": status_doc.created_at,
                        "updated_at": datetime.now().isoformat(),
                    }
                
                await self.doc_status.upsert(success_updates)
                await self._insert_done()
                logger.info(f"Successfully processed {len(success_updates)} documents")
                
            except Exception as e:
                logger.error(f"Failed during parallel processing: {str(e)}")
                
                # Update all documents to failed status if there's a general failure
                failure_updates = {}
                for doc_id, mapping in doc_chunk_mapping.items():
                    status_doc = mapping["status_doc"]
                    doc_status_id = compute_mdhash_id(status_doc.content, prefix="doc-")
                    failure_updates[doc_status_id] = {
                        "status": DocStatus.FAILED,
                        "error": str(e),
                        "content": status_doc.content,
                        "content_summary": status_doc.content_summary,
                        "content_length": status_doc.content_length,
                        "created_at": status_doc.created_at,
                        "updated_at": datetime.now().isoformat(),
                    }
                
                await self.doc_status.upsert(failure_updates)

    async def _process_entity_relation_graph(self, chunk: dict[str, Any]) -> None:
        try:
            new_kg = await extract_entities(
                chunk,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                llm_response_cache=self.llm_response_cache,
                global_config=asdict(self),
            )
            if new_kg is None:
                logger.info("No new entities or relationships extracted.")
            else:
                logger.info("New entities or relationships extracted.")
                self.chunk_entity_relation_graph = new_kg
                # Get newly extracted entities from this chunk
                if not self.addon_params["need_cross_language"]:
                    return
                extracted_entities = []
                for node in self.chunk_entity_relation_graph._graph.nodes():
                    node_data = self.chunk_entity_relation_graph._graph.nodes[node]
                    if "source_id" in node_data:
                        source_ids = node_data["source_id"].split("|") if "|" in node_data["source_id"] else [node_data["source_id"]]
                        # Check if any of the chunk keys are in source_ids
                        if any(chunk_id in chunk for chunk_id in source_ids):
                            extracted_entities.append(node)
                
                if extracted_entities:
                    # Get default language from addon_params
                    current_language = self.addon_params.get("current_language", "Vietnamese")
                    # Try to link with existing entities in other languages
                    await self._link_cross_lingual_entities(extracted_entities, current_language, matching_method=self.addon_params["matching_method"])

        except Exception as e:
            logger.error(f"Failed to extract entities and relationships: {e}")
            raise e
    
    # async def _link_cross_lingual_entities(
    #     self, 
    #     new_entities: list[str], 
    #     source_language: str = "Vietnamese",
    #     matching_method: str = "hybrid"  # Thêm tham số này
    # ) -> None:
    #     """
    #     Try to find matches for new entities in other languages and create cross-lingual links
    #     """
    #     if not new_entities:
    #         return
            
    #     logger.info(f"Looking for cross-lingual matches for {len(new_entities)} new entities")
        
    #     # Tìm các entities trong các ngôn ngữ khác
    #     existing_entities = []
    #     for node in self.chunk_entity_relation_graph._graph.nodes():
    #         if node in new_entities:
    #             continue
                
    #         node_data = self.chunk_entity_relation_graph._graph.nodes[node]
    #         entity_language = node_data.get("language", "")
            
    #         # Xác định ngôn ngữ của entity
    #         if not entity_language:
    #             for u, v, edge_data in self.chunk_entity_relation_graph._graph.edges(data=True):
    #                 if u == node or v == node:
    #                     if edge_data.get("relation_type") == "translation_equivalent":
    #                         if edge_data.get("original_language") and edge_data.get("original_language") != source_language:
    #                             entity_language = edge_data.get("original_language")
    #                             break
    #                         elif edge_data.get("translated_language") and edge_data.get("translated_language") != source_language:
    #                             entity_language = edge_data.get("translated_language")
    #                             break
            
    #         if entity_language and entity_language != source_language:
    #             existing_entities.append((node, entity_language))
        
    #     # Group entities by language
    #     entities_by_language = {}
    #     for entity, language in existing_entities:
    #         if language not in entities_by_language:
    #             entities_by_language[language] = []
    #         entities_by_language[language].append(entity)
        
    #     # Match và link cho từng ngôn ngữ
    #     for target_language, target_entities in entities_by_language.items():
    #         entity_pairs = await self._match_entities_for_linking(
    #             new_entities, 
    #             target_entities,
    #             source_language,
    #             target_language,
    #             matching_method=matching_method  # Truyền method vào
    #         )
            
    #         if entity_pairs:
    #             await self._add_cross_lingual_links(
    #                 entity_pairs, 
    #                 source_language, 
    #                 target_language
    #             )
    async def _link_cross_lingual_entities(
        self, 
        new_entities: list[str], 
        source_language: str = "Vietnamese",
        matching_method: str = "hybrid"
    ) -> None:
        if not new_entities:
            return
                
        logger.info(f"Looking for cross-lingual matches for {len(new_entities)} new entities")
        
        # Track đã match để tránh match lại
        already_matched_entities = set()
        
        # Lấy tất cả các translation links hiện có
        for u, v, edge_data in self.chunk_entity_relation_graph._graph.edges(data=True):
            if edge_data.get("relation_type") == "translation_equivalent":
                already_matched_entities.add(u)
                already_matched_entities.add(v)
        print()
        # Lọc ra các entities chưa có translation
        new_entities = [e for e in new_entities if e not in already_matched_entities]
        
        if not new_entities:
            logger.info("All entities already have translations")
            return
        
        # Tìm các entities trong các ngôn ngữ khác
        existing_entities = []
        existing_entity_languages = {}
        
        for node in self.chunk_entity_relation_graph._graph.nodes():
            if node in new_entities:
                continue
                
            node_data = self.chunk_entity_relation_graph._graph.nodes[node]
            entity_language = node_data.get("language", "")
            
            # Xác định ngôn ngữ của entity
            if not entity_language:
                entity_language = self._detect_entity_language(node)
            
            if entity_language and entity_language != source_language:
                existing_entities.append(node)
                existing_entity_languages[node] = entity_language
        
        # Group entities by language
        entities_by_language = {}
        for entity in existing_entities:
            language = existing_entity_languages[entity]
            if language not in entities_by_language:
                entities_by_language[language] = []
            entities_by_language[language].append(entity)
        
        # Match và link cho từng ngôn ngữ
        for target_language, target_entities in entities_by_language.items():
            entity_pairs = await self._match_entities_for_linking(
                new_entities, 
                target_entities,
                source_language,
                target_language,
                matching_method=matching_method
            )
            
            if entity_pairs:
                await self._add_cross_lingual_links(
                    entity_pairs, 
                    source_language, 
                    target_language
                )

    async def _detect_entity_language(self, entity: str) -> str:
        """Helper function to detect entity language from its edges"""
        for u, v, edge_data in self.chunk_entity_relation_graph._graph.edges(data=True):
            if u == entity or v == entity:
                if edge_data.get("relation_type") == "translation_equivalent":
                    if edge_data.get("original_language"):
                        return edge_data.get("original_language")
                    elif edge_data.get("translated_language"):
                        return edge_data.get("translated_language")
        return ""
    async def _match_entities_for_linking(
        self, 
        source_entities: list[str], 
        target_entities: list[str],
        source_language: str,
        target_language: str,
        similarity_threshold: float = 0.80,
        matching_method: str = "hybrid"  # Có thể là "embedding", "llm", hoặc "hybrid"
    ) -> list[tuple[str, str]]:
        """
        Match entities between languages using specified method(s)
        
        Args:
            matching_method: 
                - "embedding": Chỉ dùng embedding similarity
                - "llm": Chỉ dùng LLM
                - "hybrid": Kết hợp cả embedding và LLM
        """
        matched_pairs = []
        matched_sources = set()

        # Step 1: Exact matching (luôn thực hiện trước)
        for source_entity in source_entities:
            source_name = source_entity.strip('"').upper()
            for target_entity in target_entities:
                target_name = target_entity.strip('"').upper()
                if source_name == target_name:
                    matched_pairs.append((source_entity, target_entity))
                    matched_sources.add(source_entity)
                    break

        # Lấy các entities chưa được match
        unmatched_sources = [e for e in source_entities if e not in matched_sources]
        
        if not unmatched_sources:
            return matched_pairs

        # Step 2: Embedding-based matching
        if matching_method in ["embedding", "hybrid"] and self.embedding_func:
            embedding_matches = await self._match_entities_embedding(
                unmatched_sources,
                target_entities,
                source_language,
                target_language,
                similarity_threshold
            )
            
            for source, target in embedding_matches:
                matched_pairs.append((source, target))
                matched_sources.add(source)
                logger.info(f"Embedding matched: {source} with {target}")

        # Step 3: LLM-based matching
        if matching_method in ["llm", "hybrid"] and self.llm_model_func:
            # Chỉ xử lý các entities chưa được match bởi embedding
            still_unmatched = [e for e in unmatched_sources if e not in matched_sources]
            
            llm_matches = await self._match_entities_llm(
                still_unmatched,
                target_entities,
                source_language,
                target_language
            )
            
            matched_pairs.extend(llm_matches)

        return matched_pairs

    async def _match_entities_embedding(
        self,
        source_entities: list[str],
        target_entities: list[str],
        source_language: str,
        target_language: str,
        similarity_threshold: float = 0.90
    ) -> list[tuple[str, str]]:
        """Match entities using embedding similarity directly from vector database"""
        matches = []
        match_similarities = []  # Store similarity scores
        
        # Format entity names correctly for proper ID generation
        source_entities_formatted = [f'"{e.strip().upper()}"' if not e.strip().startswith('"') else e.strip() for e in source_entities]
        target_entities_formatted = [f'"{e.strip().upper()}"' if not e.strip().startswith('"') else e.strip() for e in target_entities]
        
        # Get entity IDs
        source_entity_ids = [compute_mdhash_id(e, prefix="ent-") for e in source_entities_formatted]
        target_entity_ids = [compute_mdhash_id(e, prefix="ent-") for e in target_entities_formatted]
        
        # Fetch embeddings directly from vector database
        source_embeddings = []
        valid_source_indices = []
        
        for i, entity_id in enumerate(source_entity_ids):
            vector = await self.entities_vdb.get_entity_embedding_by_id(entity_id)
            if vector is not None:
                source_embeddings.append(vector)
                valid_source_indices.append(i)
        
        target_embeddings = []
        valid_target_indices = []
        
        for i, entity_id in enumerate(target_entity_ids):
            vector = await self.entities_vdb.get_entity_embedding_by_id(entity_id)
            if vector is not None:
                target_embeddings.append(vector)
                valid_target_indices.append(i)
        
        if not source_embeddings or not target_embeddings:
            logger.warning("No valid entity embeddings found for comparison")
            return matches

        # Convert to numpy arrays for similarity computation
        source_embeddings_np = np.array(source_embeddings)
        target_embeddings_np = np.array(target_embeddings)
        
        # Compute cosine similarity
        similarities = cosine_similarity(source_embeddings_np, target_embeddings_np)
        
        # Find matches based on similarity threshold
        for i, source_idx in enumerate(valid_source_indices):
            max_sim_idx = similarities[i].argmax()
            if similarities[i][max_sim_idx] >= similarity_threshold:
                source_entity = source_entities[source_idx]
                target_entity = target_entities[valid_target_indices[max_sim_idx]]
                similarity_score = float(similarities[i][max_sim_idx])
                
                # Add to results for high similarity
                matches.append((source_entity, target_entity))
                match_similarities.append(similarity_score)  # Store the similarity score
                
                logger.info(
                    f"Matched entities: {source_entity} <-> {target_entity}\n"
                    f"Similarity: {similarity_score:.3f}"
                )
                
                # Get node data for logging
                source_node_data = await self.chunk_entity_relation_graph.get_node(source_entities_formatted[source_idx])
                target_node_data = await self.chunk_entity_relation_graph.get_node(target_entities_formatted[valid_target_indices[max_sim_idx]])
                
                if source_node_data and target_node_data:
                    source_desc = source_node_data.get("description", "No description")
                    target_desc = target_node_data.get("description", "No description")
                    logger.debug(
                        f"Language pair: {source_language} -> {target_language}\n"
                        f"Source description: {source_desc[:100]}...\n"
                        f"Target description: {target_desc[:100]}..."
                    )
        
        # Store matched pairs in JSON if matches were found
        if matches:
            await self._store_matched_pairs_json(
                matches,
                match_similarities,  # Pass the similarity scores
                source_entities_formatted,
                target_entities_formatted,
                valid_source_indices,
                valid_target_indices,
                source_language,
                target_language
            )
            
        return matches

    async def _store_matched_pairs_json(
        self,
        matches: list[tuple[str, str]],
        match_similarities: list[float],  # Add similarity scores parameter
        source_entities_formatted: list[str],
        target_entities_formatted: list[str],
        valid_source_indices: list[int],
        valid_target_indices: list[int],
        source_language: str,
        target_language: str
    ) -> None:
        """
        Store matched entity pairs in a JSON file
        
        Args:
            matches: List of matched entity pairs (source_entity, target_entity)
            match_similarities: List of similarity scores for each matched pair
            source_entities_formatted: Formatted source entity names
            target_entities_formatted: Formatted target entity names
            valid_source_indices: Valid indices for source entities
            valid_target_indices: Valid indices for target entities
            source_language: Source language name
            target_language: Target language name
        """
        import os
        import json
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.working_dir, "matched_pairs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"matched_pairs_{source_language}_{target_language}_{timestamp}.json")
        
        # Create dictionary to map source entities to their indices
        source_entity_idx_map = {}
        for idx, entity_idx in enumerate(valid_source_indices):
            source_entity = source_entities_formatted[entity_idx].strip('"')
            source_entity_idx_map[source_entity] = idx
            
        # Create dictionary to map target entities to their indices
        target_entity_idx_map = {}
        for idx, entity_idx in enumerate(valid_target_indices):
            target_entity = target_entities_formatted[entity_idx].strip('"')
            target_entity_idx_map[target_entity] = idx
        
        # Create list to store pairs with descriptions
        pairs_with_descriptions = []
        
        for i, (source_entity, target_entity) in enumerate(matches):
            # Get formatted entity names
            source_entity_formatted = source_entities_formatted[
                valid_source_indices[source_entity_idx_map.get(source_entity.strip('"'), 0)]
            ]
            target_entity_formatted = target_entities_formatted[
                valid_target_indices[target_entity_idx_map.get(target_entity.strip('"'), 0)]
            ]
            
            # Get entity data from graph
            source_node_data = await self.chunk_entity_relation_graph.get_node(source_entity_formatted)
            target_node_data = await self.chunk_entity_relation_graph.get_node(target_entity_formatted)
            
            # Prepare pair data with descriptions
            pair_data = {
                "original_entity": {
                    "name": source_entity.strip('"'),
                    "description": source_node_data.get("description", "No description") if source_node_data else "No description",
                    "language": source_language
                },
                "translated_entity": {
                    "name": target_entity.strip('"'),
                    "description": target_node_data.get("description", "No description") if target_node_data else "No description",
                    "language": target_language
                },
                "similarity_score": match_similarities[i]  # Use the actual similarity score
            }
            
            pairs_with_descriptions.append(pair_data)
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs_with_descriptions, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Stored {len(pairs_with_descriptions)} entity pairs in {output_file}")
        
        return output_file

    def get_matched_pairs_files(self, source_language: str = None, target_language: str = None) -> list[str]:
        """
        Get a list of all stored matched pairs JSON files, optionally filtered by language pair.
        
        Args:
            source_language: Optional filter by source language
            target_language: Optional filter by target language
            
        Returns:
            List of file paths to matched pairs JSON files
        """
        import os
        
        output_dir = os.path.join(self.working_dir, "matched_pairs")
        if not os.path.exists(output_dir):
            logger.warning("No matched pairs directory found")
            return []
            
        all_files = [
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.startswith("matched_pairs_") and f.endswith(".json")
        ]
        
        # Filter by language if specified
        if source_language and target_language:
            filtered_files = [
                f for f in all_files 
                if f"matched_pairs_{source_language}_{target_language}_" in os.path.basename(f)
            ]
            return filtered_files
        elif source_language:
            filtered_files = [
                f for f in all_files 
                if f"matched_pairs_{source_language}_" in os.path.basename(f)
            ]
            return filtered_files
        elif target_language:
            filtered_files = [
                f for f in all_files 
                if f"_{target_language}_" in os.path.basename(f)
            ]
            return filtered_files
        
        return sorted(all_files)
        
    def load_matched_pairs(self, file_path: str) -> list[dict]:
        """
        Load matched pairs data from a JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of matched pairs with descriptions
        """
        import json
        import os
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return []
            
    def get_matched_entity_pairs(
        self, 
        source_language: str = None, 
        target_language: str = None, 
        latest_only: bool = True
    ) -> list[dict]:
        """
        Convenience method to retrieve matched entity pairs
        
        Args:
            source_language: Optional filter by source language
            target_language: Optional filter by target language
            latest_only: If True, returns only pairs from the most recent file,
                         otherwise returns all pairs from all matching files
            
        Returns:
            List of matched pairs with descriptions
        """
        matched_files = self.get_matched_pairs_files(
            source_language=source_language,
            target_language=target_language
        )
        
        if not matched_files:
            logger.warning(f"No matched pairs files found for the specified languages")
            return []
            
        if latest_only:
            # Get only the most recent file
            latest_file = sorted(matched_files)[-1]
            logger.info(f"Loading matched pairs from latest file: {os.path.basename(latest_file)}")
            return self.load_matched_pairs(latest_file)
        else:
            # Combine pairs from all matching files
            all_pairs = []
            for file_path in matched_files:
                pairs = self.load_matched_pairs(file_path)
                all_pairs.extend(pairs)
            
            logger.info(f"Loaded {len(all_pairs)} pairs from {len(matched_files)} files")
            return all_pairs

    async def _match_entities_llm(
        self,
        source_entities: list[str],
        target_entities: list[str], 
        source_language: str,
        target_language: str
    ) -> list[tuple[str, str]]:
        """Match entities using LLM"""
        matches = []
        
        for source_entity in source_entities:
            source_name = source_entity.strip('"')
            prompt = f"""
            I need to match equivalent entities across languages.
            Source entity (in {source_language}): {source_name}
            
            Target entities (in {target_language}):
            {', '.join([t.strip('"') for t in target_entities])}
            
            If the source entity has an equivalent in the target entities list, return only that entity name.
            If there's no equivalent, return "NO MATCH".
            """
            
            try:
                response = await self.llm_model_func(prompt)
                response = response.strip()
                
                for target_entity in target_entities:
                    target_name = target_entity.strip('"')
                    if target_name.upper() in response.upper() and "NO MATCH" not in response.upper():
                        matches.append((source_entity, target_entity))
                        logger.info(f"LLM matched: {source_name} with {target_name}")
                        break
            except Exception as e:
                logger.error(f"Error using LLM for entity matching: {e}")
                
        return matches
    async def _insert_done(self) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.text_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

    # async def _add_cross_lingual_links(
    #     self, 
    #     entity_pairs: list[tuple[str, str]],
    #     source_language: str,
    #     target_language: str
    # ) -> None:
    #     """
    #     Add cross-lingual links between matched entity pairs
        
    #     Args:
    #         entity_pairs: List of matched entity pairs (source_entity, target_entity)
    #         source_language: Source language name
    #         target_language: Target language name
    #     """
    #     for source_entity, target_entity in entity_pairs:
    #         # Check if entities still exist (they might have been deleted)
    #         if (await self.chunk_entity_relation_graph.has_node(source_entity) and 
    #             await self.chunk_entity_relation_graph.has_node(target_entity)):
                
    #             # Check if they already have a TRANSLATED relationship
    #             if await self.chunk_entity_relation_graph.has_edge(source_entity, target_entity):
    #                 edge_data = await self.chunk_entity_relation_graph.get_edge(source_entity, target_entity)
    #                 if edge_data and edge_data.get("relation_type") == "translation_equivalent":
    #                     logger.info(f"Entities {source_entity} and {target_entity} already have a translation relationship")
    #                     continue
                
    #             # Create cross-lingual edges based on the pattern seen in the example GraphML
    #             edge_data = {
    #                 "weight": 1.0,
    #                 "description": f"Translation equivalent ({source_language} ↔ {target_language})",
    #                 "keywords": f"translation,cross-lingual,{source_language},{target_language}",
    #                 "relation_type": "translation_equivalent",
    #                 "languages": f"{source_language},{target_language}",
    #                 "source_id": "cross_lingual",
    #                 "original_language": source_language,
    #                 "translated_language": target_language
    #             }
                
    #             # Add edges in both directions with the same data
    #             await self.chunk_entity_relation_graph.upsert_edge(
    #                 source_entity, target_entity, edge_data
    #             )
                
    #             logger.info(f"Created cross-lingual link: {source_entity} <-> {target_entity}")

    async def _add_cross_lingual_links(
        self, 
        entity_pairs: list[tuple[str, str]],
        source_language: str,
        target_language: str
    ) -> None:
        """
        Add cross-lingual links between matched entity pairs
        
        Args:
            entity_pairs: List of matched entity pairs (source_entity, target_entity)
            source_language: Source language name
            target_language: Target language name
        """
        for source_entity, target_entity in entity_pairs:
            # Check if entities still exist (they might have been deleted)
            if (await self.chunk_entity_relation_graph.has_node(source_entity) and 
                await self.chunk_entity_relation_graph.has_node(target_entity)):
                
                # Cải tiến 6: Kiểm tra trùng lặp trước khi tạo liên kết
                if await self.chunk_entity_relation_graph.has_edge(source_entity, target_entity):
                    edge_data = await self.chunk_entity_relation_graph.get_edge(source_entity, target_entity)
                    if edge_data and edge_data.get("relation_type") == "translation_equivalent":
                        # Cải tiến 5: Thêm logging chi tiết
                        logger.info(f"Entities {source_entity} and {target_entity} already have a translation relationship")
                        continue
                
                # Cải tiến 4: Edge data với metadata phong phú hơn
                edge_data = {
                    "weight": 1.0,
                    "description": f"Translation equivalent ({source_language} ↔ {target_language})",
                    "keywords": f"translation,cross-lingual,{source_language},{target_language}",
                    "relation_type": "translation_equivalent",
                    "languages": f"{source_language},{target_language}",
                    "source_id": "cross_lingual",
                    "original_language": source_language,
                    "translated_language": target_language,
                    "created_at": datetime.now().isoformat(),  # Thêm timestamp
                    "confidence_score": 1.0,  # Có thể thay đổi dựa trên phương pháp matching
                    "matching_method": self.addon_params.get("matching_method", "hybrid")  # Lưu phương pháp matching đã sử dụng
                }
                
                # Add edges in both directions with the same data
                try:
                    await self.chunk_entity_relation_graph.upsert_edge(
                        source_entity, target_entity, edge_data
                    )
                    # Cải tiến 5: Thêm logging chi tiết
                    logger.info(f"Created cross-lingual link: {source_entity} <-> {target_entity}")
                    logger.debug(f"Edge data: {edge_data}")
                except Exception as e:
                    # Cải tiến 5: Log lỗi chi tiết
                    logger.error(f"Failed to create cross-lingual link between {source_entity} and {target_entity}: {e}")
                    continue

                # Cập nhật metadata của nodes để đánh dấu chúng đã được liên kết
                try:
                    # Update source node
                    source_node_data = await self.chunk_entity_relation_graph.get_node(source_entity)
                    if source_node_data:
                        source_node_data["has_translations"] = True
                        source_node_data["translated_languages"] = source_node_data.get("translated_languages", "") + f"|{target_language}"
                        await self.chunk_entity_relation_graph.upsert_node(source_entity, source_node_data)

                    # Update target node
                    target_node_data = await self.chunk_entity_relation_graph.get_node(target_entity)
                    if target_node_data:
                        target_node_data["has_translations"] = True
                        target_node_data["translated_languages"] = target_node_data.get("translated_languages", "") + f"|{source_language}"
                        await self.chunk_entity_relation_graph.upsert_node(target_entity, target_node_data)
                    
                    # Cải tiến 5: Log thành công
                    logger.debug(f"Updated metadata for nodes {source_entity} and {target_entity}")
                except Exception as e:
                    # Cải tiến 5: Log lỗi chi tiết
                    logger.error(f"Failed to update node metadata: {e}")

    def insert_custom_kg(self, custom_kg: dict[str, Any]) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_custom_kg(custom_kg))

    async def ainsert_custom_kg(self, custom_kg: dict[str, Any]) -> None:
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
            for chunk_data in custom_kg.get("chunks", {}):
                chunk_content = chunk_data["content"]
                source_id = chunk_data["source_id"]
                chunk_id = compute_mdhash_id(chunk_content.strip(), prefix="chunk-")

                chunk_entry = {"content": chunk_content.strip(), "source_id": source_id}
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if all_chunks_data:
                await self.chunks_vdb.upsert(all_chunks_data)
            if all_chunks_data:
                await self.text_chunks.upsert(all_chunks_data)

            # Insert entities into knowledge graph
            all_entities_data: list[dict[str, str]] = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = f'"{entity_data["entity_name"].upper()}"'
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                # source_id = entity_data["source_id"]
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Detect language from entity name
                language = entity_data.get("language", detect_language(entity_name))

                # Prepare node data
                node_data: dict[str, str] = {
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                    "language": language,
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data: list[dict[str, str]] = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = f'"{relationship_data["src_id"].upper()}"'
                tgt_id = f'"{relationship_data["tgt_id"].upper()}"'
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                # source_id = relationship_data["source_id"]
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")

                # Detect language from description
                language = relationship_data.get("language", detect_language(description))

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                                "language": language,
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                        "language": language,
                    },
                )
                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "language": language,
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage if needed
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
                    "language": dp.get("language", "Unknown"),
                }
                for dp in all_entities_data
            }
            await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage if needed
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "content": dp["keywords"]
                    + dp["src_id"]
                    + dp["tgt_id"]
                    + dp["description"],
                    "language": dp.get("language", "Unknown"),
                }
                for dp in all_relationships_data
            }
            await self.relationships_vdb.upsert(data_for_vdb)

        finally:
            if update_storage:
                await self._insert_done()

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, system_prompt))  # type: ignore

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                system_prompt=system_prompt,
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response


    def retrieval(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aretrieval(query, param, system_prompt))  # type: ignore

    async def aretrieval(
        self,
        query: str,
        param: QueryParam = QueryParam(only_need_context=True),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_retrieval(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
                system_prompt=system_prompt,
            )
        elif param.mode == "naive":
            response = await naive_retrieval(
                query,
                param,
                asdict(self),
                self.chunks_vdb,
                self.text_chunks
            )
        else:
            raise ValueError(f"Unsupport mode {param.mode}")
        await self._query_done()
        return response


    def query_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ):
        """
        1. Extract keywords from the 'query' using new function in operate.py.
        2. Then run the standard aquery() flow with the final prompt (formatted_question).
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_separate_keyword_extraction(query, prompt, param)
        )

    async def aquery_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ) -> str | AsyncIterator[str]:
        """
        1. Calls extract_keywords_only to get HL/LL keywords from 'query'.
        2. Then calls kg_query(...) or naive_query(...), etc. as the main query, while also injecting the newly extracted keywords if needed.
        """
        # ---------------------
        # STEP 1: Keyword Extraction
        # ---------------------
        # We'll assume 'extract_keywords_only(...)' returns (hl_keywords, ll_keywords).
        hl_keywords, ll_keywords = await extract_keywords_only(
            text=query,
            param=param,
            global_config=asdict(self),
            hashing_kv=self.llm_response_cache
            or self.key_string_value_json_storage_cls(
                namespace=make_namespace(
                    self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                ),
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            ),
        )

        param.hl_keywords = hl_keywords
        param.ll_keywords = ll_keywords

        # ---------------------
        # STEP 2: Final Query Logic
        # ---------------------

        # Create a new string with the prompt and the keywords
        ll_keywords_str = ", ".join(ll_keywords)
        hl_keywords_str = ", ".join(hl_keywords)
        formatted_question = f"{prompt}\n\n### Keywords:\nHigh-level: {hl_keywords_str}\nLow-level: {ll_keywords_str}\n\n### Query:\n{query}"

        if param.mode in ["local", "global", "hybrid"]:
            response = await kg_query_with_keywords(
                formatted_question,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        elif param.mode == "naive":
            response = await naive_query(
                formatted_question,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        elif param.mode == "mix":
            response = await mix_kg_vector_query(
                formatted_question,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                param,
                asdict(self),
                hashing_kv=self.llm_response_cache
                if self.llm_response_cache
                and hasattr(self.llm_response_cache, "global_config")
                else self.key_string_value_json_storage_cls(
                    namespace=make_namespace(
                        self.namespace_prefix, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
                    ),
                    global_config=asdict(self),
                    embedding_func=self.embedding_func,
                ),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")

        await self._query_done()
        return response

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    def delete_by_entity(self, entity_name: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str) -> None:
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_vdb.delete_entity(entity_name)
            await self.relationships_vdb.delete_entity_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relationships have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self) -> None:
        await asyncio.gather(
            *[
                cast(StorageNameSpace, storage_inst).index_done_callback()
                for storage_inst in [  # type: ignore
                    self.entities_vdb,
                    self.relationships_vdb,
                    self.chunk_entity_relation_graph,
                ]
            ]
        )

    def _get_content_summary(self, content: str, max_length: int = 100) -> str:
        """Get summary of document content

        Args:
            content: Original document content
            max_length: Maximum length of summary

        Returns:
            Truncated content with ellipsis if needed
        """
        content = content.strip()
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            # 1. Get the document status and related data
            doc_status = await self.doc_status.get_by_id(doc_id)
            if not doc_status:
                logger.warning(f"Document {doc_id} not found")
                return

            logger.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all related chunks
            chunks = await self.text_chunks.get_by_id(doc_id)
            if not chunks:
                return

            chunk_ids = list(chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # 3. Before deleting, check the related entities and relationships for these chunks
            for chunk_id in chunk_ids:
                # Check entities
                entities = [
                    dp
                    for dp in self.entities_vdb.client_storage["data"]
                    if dp.get("source_id") == chunk_id
                ]
                logger.debug(f"Chunk {chunk_id} has {len(entities)} related entities")

                # Check relationships
                relations = [
                    dp
                    for dp in self.relationships_vdb.client_storage["data"]
                    if dp.get("source_id") == chunk_id
                ]
                logger.debug(f"Chunk {chunk_id} has {len(relations)} related relations")

            # Continue with the original deletion process...

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 5. Find and process entities and relationships that have these chunks as source
            # Get all nodes in the graph
            nodes = self.chunk_entity_relation_graph._graph.nodes(data=True)
            edges = self.chunk_entity_relation_graph._graph.edges(data=True)

            # Track which entities and relationships need to be deleted or updated
            entities_to_delete = set()
            entities_to_update = {}  # entity_name -> new_source_id
            relationships_to_delete = set()
            relationships_to_update = {}  # (src, tgt) -> new_source_id

            # Process entities
            for node, data in nodes:
                if "source_id" in data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        entities_to_delete.add(node)
                        logger.debug(
                            f"Entity {node} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        entities_to_update[node] = new_source_id
                        logger.debug(
                            f"Entity {node} will be updated with new source_id: {new_source_id}"
                        )

            # Process relationships
            for src, tgt, data in edges:
                if "source_id" in data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        relationships_to_delete.add((src, tgt))
                        logger.debug(
                            f"Relationship {src}-{tgt} marked for deletion - no remaining sources"
                        )
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        relationships_to_update[(src, tgt)] = new_source_id
                        logger.debug(
                            f"Relationship {src}-{tgt} will be updated with new source_id: {new_source_id}"
                        )

            # Delete entities
            if entities_to_delete:
                for entity in entities_to_delete:
                    await self.entities_vdb.delete_entity(entity)
                    logger.debug(f"Deleted entity {entity} from vector DB")
                self.chunk_entity_relation_graph.remove_nodes(list(entities_to_delete))
                logger.debug(f"Deleted {len(entities_to_delete)} entities from graph")

            # Update entities
            for entity, new_source_id in entities_to_update.items():
                node_data = self.chunk_entity_relation_graph._graph.nodes[entity]
                node_data["source_id"] = new_source_id
                await self.chunk_entity_relation_graph.upsert_node(entity, node_data)
                logger.debug(
                    f"Updated entity {entity} with new source_id: {new_source_id}"
                )

            # Delete relationships
            if relationships_to_delete:
                for src, tgt in relationships_to_delete:
                    rel_id_0 = compute_mdhash_id(src + tgt, prefix="rel-")
                    rel_id_1 = compute_mdhash_id(tgt + src, prefix="rel-")
                    await self.relationships_vdb.delete([rel_id_0, rel_id_1])
                    logger.debug(f"Deleted relationship {src}-{tgt} from vector DB")
                self.chunk_entity_relation_graph.remove_edges(
                    list(relationships_to_delete)
                )
                logger.debug(
                    f"Deleted {len(relationships_to_delete)} relationships from graph"
                )

            # Update relationships
            for (src, tgt), new_source_id in relationships_to_update.items():
                edge_data = self.chunk_entity_relation_graph._graph.edges[src, tgt]
                edge_data["source_id"] = new_source_id
                await self.chunk_entity_relation_graph.upsert_edge(src, tgt, edge_data)
                logger.debug(
                    f"Updated relationship {src}-{tgt} with new source_id: {new_source_id}"
                )

            # 6. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            # 7. Ensure all indexes are updated
            await self._insert_done()

            logger.info(
                f"Successfully deleted document {doc_id} and related data. "
                f"Deleted {len(entities_to_delete)} entities and {len(relationships_to_delete)} relationships. "
                f"Updated {len(entities_to_update)} entities and {len(relationships_to_update)} relationships."
            )

            # Add verification step
            async def verify_deletion():
                # Verify if the document has been deleted
                if await self.full_docs.get_by_id(doc_id):
                    logger.error(f"Document {doc_id} still exists in full_docs")

                # Verify if chunks have been deleted
                remaining_chunks = await self.text_chunks.get_by_id(doc_id)
                if remaining_chunks:
                    logger.error(f"Found {len(remaining_chunks)} remaining chunks")

                # Verify entities and relationships
                for chunk_id in chunk_ids:
                    # Check entities
                    entities_with_chunk = [
                        dp
                        for dp in self.entities_vdb.client_storage["data"]
                        if chunk_id
                        in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                    ]
                    if entities_with_chunk:
                        logger.error(
                            f"Found {len(entities_with_chunk)} entities still referencing chunk {chunk_id}"
                        )

                    # Check relationships
                    relations_with_chunk = [
                        dp
                        for dp in self.relationships_vdb.client_storage["data"]
                        if chunk_id
                        in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                    ]
                    if relations_with_chunk:
                        logger.error(
                            f"Found {len(relations_with_chunk)} relations still referencing chunk {chunk_id}"
                        )

            await verify_deletion()

        except Exception as e:
            logger.error(f"Error while deleting document {doc_id}: {e}")

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity

        Args:
            entity_name: Entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing entity information, including:
                - entity_name: Entity name
                - source_id: Source document ID
                - graph_data: Complete node data from the graph database
                - vector_data: (optional) Data from the vector database
        """
        entity_name = f'"{entity_name.upper()}"'

        # Get information from the graph
        node_data = await self.chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        result: dict[str, str | None | dict[str, str]] = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": node_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = self.entities_vdb._client.get([entity_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship

        Args:
            src_entity: Source entity name (no need for quotes)
            tgt_entity: Target entity name (no need for quotes)
            include_vector_data: Whether to include data from the vector database

        Returns:
            dict: A dictionary containing relationship information, including:
                - src_entity: Source entity name
                - tgt_entity: Target entity name
                - source_id: Source document ID
                - graph_data: Complete edge data from the graph database
                - vector_data: (optional) Data from the vector database
        """
        src_entity = f'"{src_entity.upper()}"'
        tgt_entity = f'"{tgt_entity.upper()}"'

        # Get information from the graph
        edge_data = await self.chunk_entity_relation_graph.get_edge(
            src_entity, tgt_entity
        )
        source_id = edge_data.get("source_id") if edge_data else None

        result: dict[str, str | None | dict[str, str]] = {
            "src_entity": src_entity,
            "tgt_entity": tgt_entity,
            "source_id": source_id,
            "graph_data": edge_data,
        }

        # Optional: Get vector database information
        if include_vector_data:
            rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
            vector_data = self.relationships_vdb._client.get([rel_id])
            result["vector_data"] = vector_data[0] if vector_data else None

        return result

    def set_language(self, language: str) -> None:
        """Set the language for prompts
        
        Args:
            language: Language code ('EN' for English, 'VI' for Vietnamese)
        """
        if language not in ["English", "Vietnamese"]:
            logger.warning(f"Unsupported language: {language}. Falling back to English.")
            language = "EN"
        
        self.addon_params["language"] = language
        logger.info(f"Language set to: {language}")

    async def process_document_pairs_in_batches(
        self,
        document_pairs_data: List[List[str]],
        source_language: str = "Vietnamese",
        target_language: str = "English",
        store_translations: bool = True,
        translation_db_path: Optional[str] = None,
        show_progress: bool = True
    ):
        """
        Process document pairs in batches with controlled concurrency.
        
        Args:
            document_pairs_data: List of pairs where each pair is [original_text, translated_text]
            batch_size: Number of document pairs to process concurrently in each batch
            source_language: Source language of original texts
            target_language: Target language of translated texts
            store_translations: Whether to store entity and relation translations
            translation_db_path: Path to store translation mappings
            show_progress: Whether to show a progress bar
            
        Returns:
            List of tuples with (original_doc_id, translated_doc_id) for each pair
        """
        batch_size = self.addon_params.get("insert_batch_size", 10)
        all_results = []
        total_pairs = len(document_pairs_data)
        processed = 0
        
        # Optional progress bar
        pbar = None
        if show_progress:
            try:
                pbar = tqdm(total=total_pairs, desc="Processing document pairs")
            except:
                show_progress = False
                logger.warning("tqdm not available for progress display")
        
        # Process in batches
        for i in range(0, total_pairs, batch_size):
            batch = document_pairs_data[i:i+batch_size]
            batch_tasks = []
            
            # Create tasks for this batch
            for pair in batch:
                original_text, translated_text = pair
                task = self.ainsert_duo(
                    data_original=original_text,
                    data_translated=translated_text,
                    source_language=source_language,
                    target_language=target_language,
                    store_translations=store_translations,
                    translation_db_path=translation_db_path
                )
                batch_tasks.append(task)
            
            # Process this batch concurrently
            try:
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_pairs + batch_size - 1)//batch_size} with {len(batch)} document pairs")
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results and exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing document pair {i+j}: {result}")
                        all_results.append(None)
                    else:
                        all_results.append(result)
                
                processed += len(batch)
                if pbar:
                    pbar.update(len(batch))
                
                logger.info(f"Completed batch {i//batch_size + 1}: {processed}/{total_pairs} pairs processed")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add None results for this batch
                all_results.extend([None] * len(batch))
                if pbar:
                    pbar.update(len(batch))
        
        if pbar:
            pbar.close()
        
        # Log final results
        successful = sum(1 for r in all_results if r is not None)
        logger.info(f"Overall processed {successful} out of {total_pairs} document pairs successfully")
        
        return all_results
    
# new_func
    def insert_duo(
        self,
        document_pairs_data: List[List[str]],
        source_language="Vietnamese",
        target_language="English",
        store_translations=True,
        translation_db_path=None
    ):
        """
        Insert a document in both its original language and translated version.
        
        Args:
            data_original: Original document data (Vietnamese)
            data_translated: Translated document data (if None, will be generated using LLM)
            source_language: Source language (default: "Vietnamese")
            target_language: Target language (default: "English")
            store_translations: Whether to store entity and relation translations
            translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
        
        Returns:
            Tuple of (original_doc_id, translated_doc_id)
        """
        # loop = always_get_an_event_loop()
        # return loop.run_until_complete(
        #     self.ainsert_duo(data_original, data_translated, source_language, target_language, store_translations, translation_db_path)
        # )

        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.process_document_pairs_in_batches(
                document_pairs_data,
                source_language=source_language,
                target_language=target_language,
                store_translations=store_translations,
                translation_db_path=translation_db_path

            )
        )

## old version

    # async def ainsert_duo(
    #     self,
    #     data_original,
    #     data_translated=None,
    #     source_language="Vietnamese",
    #     target_language="English",
    #     store_translations=True,
    #     translation_db_path=None
    # ):
    #     """
    #     Async insert a document in both its original language and translated version.
    #     Ensures exact 1:1 mapping between entities and relationships.
        
    #     Args:
    #         data_original: Original document data (Vietnamese)
    #         data_translated: Translated document data (if None, will be generated using LLM)
    #         source_language: Source language (default: "Vietnamese")
    #         target_language: Target language (default: "English")
    #         store_translations: Whether to store entity and relation translations
    #         translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
        
    #     Returns:
    #         Tuple of (original_doc_id, translated_doc_id)
    #     """
    #     if translation_db_path is None:
    #         translation_db_path = os.path.join(self.working_dir, "translations.json")
        
    #     logger.info(f"Starting duo insertion: {source_language} and {target_language}")
        
    #     # First process the original document
    #     original_doc_id = compute_mdhash_id(data_original.strip(), prefix="doc-")
    #     # original_doc_id = compute_mdhash_id(data_original.strip(), prefix="doc-")
        
    #     # Kiểm tra nếu văn bản đã tồn tại trong doc_status và đã PROCESSED
    #     doc_exists = await self.doc_status.get_by_id(original_doc_id)
    #     if doc_exists and doc_exists.get("status") == DocStatus.PROCESSED:
    #         translated_doc_id = compute_mdhash_id(data_translated.strip(), prefix="doc-")
    #         logger.info(f"Document {original_doc_id} already processed, skipping duo insertion")
    #         return original_doc_id, translated_doc_id
    #     # Store original document in doc status first
    #     await self.doc_status.upsert({
    #         original_doc_id: {
    #             "content": data_original,
    #             "content_summary": self._get_content_summary(data_original),
    #             "content_length": len(data_original),
    #             "status": DocStatus.PENDING,
    #             "language": source_language,
    #             "created_at": datetime.now().isoformat(),
    #             "updated_at": datetime.now().isoformat(),
    #         }
    #     })
        
    #     # Store data to full_docs
    #     await self.full_docs.upsert({original_doc_id: {"content": data_original.strip()}})
        
    #     # Create chunks for original document
    #     original_chunks = {
    #         compute_mdhash_id(dp["content"], prefix="chunk-"): {
    #             **dp,
    #             "full_doc_id": original_doc_id,
    #         }
    #         for dp in self.chunking_func(
    #             data_original,
    #             None,
    #             False,
    #             self.chunk_overlap_token_size,
    #             self.chunk_token_size,
    #             self.tiktoken_model_name,
    #         )
    #     }
        
    #     # Process the chunks and extract entities/relations
    #     # Insert chunks to vector storage and text chunks storage
    #     await asyncio.gather(
    #         self.chunks_vdb.upsert(original_chunks),
    #         self.text_chunks.upsert(original_chunks),
    #     )
        
    #     # Extract entities and relations from original document
    #     logger.info(f"Extracting entities and relations from {source_language} document")
    #     original_extraction_result = await extract_entities(
    #         original_chunks,
    #         knowledge_graph_inst=self.chunk_entity_relation_graph,
    #         entity_vdb=self.entities_vdb,
    #         relationships_vdb=self.relationships_vdb,
    #         llm_response_cache=self.llm_response_cache,
    #         global_config=asdict(self),
    #     )
        
    #     # Get entities and relations from the original document
    #     original_entities = await self._get_document_entities(original_doc_id, original_chunks)
    #     original_relations = await self._get_document_relations(original_doc_id, original_chunks)
        
    #     logger.info(f"Found {len(original_entities)} entities and {len(original_relations)} relations in {source_language} document")
        
    #     # Update status for original document
    #     await self.doc_status.upsert({
    #         original_doc_id: {
    #             "status": DocStatus.PROCESSED,
    #             "chunks_count": len(original_chunks),
    #             "content": data_original,
    #             "content_summary": self._get_content_summary(data_original),
    #             "content_length": len(data_original),
    #             "updated_at": datetime.now().isoformat(),
    #         }
    #     })
        
    #     # If translated data is not provided, generate it
    #     if data_translated is None or data_translated == "":
    #         logger.info(f"Generating translation for document in {target_language}")
    #         data_translated = await self._translate_preserving_structure(
    #             data_original, 
    #             source_language,
    #             target_language
    #         )
        
    #     # Process translated document
    #     translated_doc_id = compute_mdhash_id(data_translated.strip(), prefix="doc-")
        
    #     # Store translated document in doc status
    #     await self.doc_status.upsert({
    #         translated_doc_id: {
    #             "content": data_translated,
    #             "content_summary": self._get_content_summary(data_translated),
    #             "content_length": len(data_translated),
    #             "status": DocStatus.PENDING,
    #             "language": target_language,
    #             "created_at": datetime.now().isoformat(),
    #             "updated_at": datetime.now().isoformat(),
    #         }
    #     })
        
    #     # Store data for translated document
    #     await self.full_docs.upsert({translated_doc_id: {"content": data_translated.strip()}})
        
    #     # Create chunks for translated document
    #     translated_chunks = {
    #         compute_mdhash_id(dp["content"], prefix="chunk-"): {
    #             **dp,
    #             "full_doc_id": translated_doc_id,
    #         }
    #         for dp in self.chunking_func(
    #             data_translated,
    #             None,
    #             False,
    #             self.chunk_overlap_token_size,
    #             self.chunk_token_size,
    #             self.tiktoken_model_name,
    #         )
    #     }
        
    #     # Insert chunks to vector storage and text chunks storage
    #     await asyncio.gather(
    #         self.chunks_vdb.upsert(translated_chunks),
    #         self.text_chunks.upsert(translated_chunks),
    #     )
        
    #     # Extract corresponding entities and relations in the translated document
    #     # using the original entities as a guide
    #     logger.info(f"Extracting matching entities and relations from {target_language} document")
        
    #     # First pass: Extract matching entities - ONLY EXTRACT, DON'T SAVE YET
    #     translated_entities = await self._extract_matching_entities(
    #         original_entities,
    #         data_translated,
    #         source_language,
    #         target_language,
    #         translated_chunks
    #     )
        
    #     # Second pass: Extract matching relations - ONLY EXTRACT, DON'T SAVE YET
    #     translated_relations = await self._extract_matching_relations(
    #         original_relations,
    #         translated_entities,
    #         data_translated,
    #         source_language,
    #         target_language,
    #         translated_chunks
    #     )
        
    #     logger.info(f"Extracted {len(translated_entities)} entities and {len(translated_relations)} relations in {target_language} document")
        
    #     # Verify the counts match
    #     if len(original_entities) != len(translated_entities):
    #         logger.warning(f"Entity count mismatch: {len(original_entities)} {source_language} vs {len(translated_entities)} {target_language}")
    #         # Force entity count to match by requesting a fix
    #         translated_entities = await self._fix_entity_count_mismatch(
    #             original_entities,
    #             translated_entities,
    #             data_translated,
    #             source_language,
    #             target_language,
    #             translated_chunks
    #         )
        
    #     if len(original_relations) != len(translated_relations):
    #         logger.warning(f"Relation count mismatch: {len(original_relations)} {source_language} vs {len(translated_relations)} {target_language}")
    #         # Force relation count to match by requesting a fix
    #         translated_relations = await self._fix_relation_count_mismatch(
    #             original_relations,
    #             translated_relations, 
    #             translated_entities,
    #             data_translated,
    #             source_language, 
    #             target_language,
    #             translated_chunks
    #         )
        
    #     # NOW THAT WE HAVE VERIFIED ENTITIES AND RELATIONS MATCH, SAVE THEM TO THE GRAPH
    #     logger.info(f"Saving verified entities and relations to knowledge graph")
        
    #     nodes_data_map = {}
    #     for entity in translated_entities:
    #         entity_name = f'"{entity["name"].upper()}"'
            
    #         # Get first chunk ID for this document
    #         chunk_id = next(iter(translated_chunks.keys()))
            
    #         # Chuẩn bị data
    #         if entity_name not in nodes_data_map:
    #             nodes_data_map[entity_name] = []
            
    #         nodes_data_map[entity_name].append({
    #             "entity_type": f'"{entity["type"].upper()}"',
    #             "description": entity["description"],
    #             "source_id": chunk_id,
    #             "language": target_language,
    #         })

    #     # Chuẩn bị dữ liệu cho edges
    #     edges_data_map = {}
    #     for relation in translated_relations:
    #         src_entity = f'"{relation["source"].upper()}"'
    #         tgt_entity = f'"{relation["target"].upper()}"'
            
    #         # Get first chunk ID for this document
    #         chunk_id = next(iter(translated_chunks.keys()))
            
    #         edge_key = (src_entity, tgt_entity)
    #         if edge_key not in edges_data_map:
    #             edges_data_map[edge_key] = []
            
    #         edges_data_map[edge_key].append({
    #             "description": relation["description"],
    #             "keywords": relation["keywords"],
    #             "weight": 1.0,
    #             "source_id": chunk_id,
    #             "language": target_language,
    #         })
    #     # 2. Thực hiện merge và upsert song song
    #     all_entities_tasks = [
    #         _merge_nodes_then_upsert(entity_name, nodes_data, 
    #                             self.chunk_entity_relation_graph, asdict(self))
    #         for entity_name, nodes_data in nodes_data_map.items()
    #     ]

    #     all_edges_tasks = [
    #         _merge_edges_then_upsert(src_id, tgt_id, edges_data,
    #                             self.chunk_entity_relation_graph, asdict(self))
    #         for (src_id, tgt_id), edges_data in edges_data_map.items()
    #     ]

    #     # Thực thi song song
    #     all_entities_data = await asyncio.gather(*all_entities_tasks)
    #     all_relationships_data = await asyncio.gather(*all_edges_tasks)

    #     # 3. Chuẩn bị dữ liệu cho vector databases
    #     entities_vdb_data = {
    #         compute_mdhash_id(entity_data["entity_name"], prefix="ent-"): {
    #             "content": f"{entity_data['entity_name']} {entity_data['description']}",
    #             "entity_name": entity_data["entity_name"],
    #             "language": entity_data.get("language", target_language),
    #         }
    #         for entity_data in all_entities_data
    #     }

    #     relationships_vdb_data = {
    #         compute_mdhash_id(rel_data["src_id"] + rel_data["tgt_id"], prefix="rel-"): {
    #             "content": f"{rel_data['keywords']} {rel_data['src_id']} {rel_data['tgt_id']} {rel_data['description']}",
    #             "src_id": rel_data["src_id"],
    #             "tgt_id": rel_data["tgt_id"],
    #             "language": rel_data.get("language", target_language),
    #         }
    #         for rel_data in all_relationships_data
    #     }

    #     # 4. Thực hiện insert vector DB song song
    #     await asyncio.gather(
    #         self.entities_vdb.upsert(entities_vdb_data),
    #         self.relationships_vdb.upsert(relationships_vdb_data)
    #     )
    #     # Update status for translated document
    #     await self.doc_status.upsert({
    #         translated_doc_id: {
    #             "status": DocStatus.PROCESSED,
    #             "chunks_count": len(translated_chunks),
    #             "content": data_translated,
    #             "content_summary": self._get_content_summary(data_translated),
    #             "content_length": len(data_translated),
    #             "updated_at": datetime.now().isoformat(),
    #         }
    #     })
        
    #     # Create cross-lingual edges between entities
    #     logger.info(f"Creating cross-lingual edges between {source_language} and {target_language} entities")
    #     await self._create_cross_lingual_edges(
    #         original_entities,
    #         translated_entities,
    #         source_language,
    #         target_language
    #     )
        
    #     # Store translation pairs if requested
    #     if store_translations:
    #         logger.info(f"Storing translation pairs to {translation_db_path}")
    #         await self._store_translation_pairs(
    #             original_entities,
    #             translated_entities,
    #             original_relations,
    #             translated_relations,
    #             source_language,
    #             target_language,
    #             translation_db_path
    #         )
        
    #     # Save changes to all storages
    #     await self._insert_done()
        
    #     return original_doc_id, translated_doc_id
### new_version
    

    async def ainsert_duo(
        self: LightRAG,
        data_original,
        data_translated=None,
        source_language="Vietnamese",
        target_language="English",
        store_translations=True,
        translation_db_path=None
    ):
        """
        Async insert a document in both its original language and translated version.
        Ensures exact 1:1 mapping between entities and relationships with maximized parallel execution.
        
        Args:
            data_original: Original document data (Vietnamese)
            data_translated: Translated document data (if None, will be generated using LLM)
            source_language: Source language (default: "Vietnamese")
            target_language: Target language (default: "English")
            store_translations: Whether to store entity and relation translations
            translation_db_path: Path to store translation mappings (defaults to working_dir/translations.json)
        
        Returns:
            Tuple of (original_doc_id, translated_doc_id)
        """
        if translation_db_path is None:
            translation_db_path = os.path.join(self.working_dir, "translations.json")
        
        logger.info(f"Starting duo insertion: {source_language} and {target_language}")
        
        # Compute document IDs for original and translated docs
        original_doc_id = compute_mdhash_id(data_original.strip(), prefix="doc-")
        
        # Check if document already exists and is processed
        doc_exists = await self.doc_status.get_by_id(original_doc_id)
        if doc_exists and doc_exists.get("status") == DocStatus.PROCESSED:
            translated_doc_id = compute_mdhash_id((data_translated or "").strip(), prefix="doc-")
            logger.info(f"Document {original_doc_id} already processed, skipping duo insertion")
            return original_doc_id, translated_doc_id
        
        # If translated data is not provided, generate it
        if data_translated is None or data_translated == "":
            logger.info(f"Generating translation for document in {target_language}")
            data_translated = await self._translate_preserving_structure(
                data_original, 
                source_language,
                target_language
            )
        
        # Compute translated document ID
        translated_doc_id = compute_mdhash_id(data_translated.strip(), prefix="doc-")
        
        # Create chunks for both original and translated documents
        original_chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": original_doc_id,
            }
            for dp in self.chunking_func(
                data_original,
                None,
                False,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name,
            )
        }
        
        translated_chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": translated_doc_id,
            }
            for dp in self.chunking_func(
                data_translated,
                None,
                False,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
                self.tiktoken_model_name,
            )
        }
        
        # Initialize all document metadata and storage operations in parallel
        init_tasks = [
            # Store document metadata
            self.doc_status.upsert({
                original_doc_id: {
                    "content": data_original,
                    "content_summary": self._get_content_summary(data_original),
                    "content_length": len(data_original),
                    "status": DocStatus.PENDING,
                    "language": source_language,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
            }),
            self.doc_status.upsert({
                translated_doc_id: {
                    "content": data_translated,
                    "content_summary": self._get_content_summary(data_translated),
                    "content_length": len(data_translated),
                    "status": DocStatus.PENDING,
                    "language": target_language,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
            }),
            # Store full document content
            self.full_docs.upsert({original_doc_id: {"content": data_original.strip()}}),
            self.full_docs.upsert({translated_doc_id: {"content": data_translated.strip()}}),
            # Store chunks in vector and text storage
            self.chunks_vdb.upsert(original_chunks),
            self.text_chunks.upsert(original_chunks),
            self.chunks_vdb.upsert(translated_chunks),
            self.text_chunks.upsert(translated_chunks),
        ]
        
        # Execute all initialization tasks in parallel
        await asyncio.gather(*init_tasks)
        
        # Update status to processing
        processing_tasks = [
            self.doc_status.upsert({
                original_doc_id: {
                    "status": DocStatus.PROCESSING,
                    "updated_at": datetime.now().isoformat(),
                }
            }),
            self.doc_status.upsert({
                translated_doc_id: {
                    "status": DocStatus.PROCESSING,
                    "updated_at": datetime.now().isoformat(),
                }
            })
        ]
        
        await asyncio.gather(*processing_tasks)
        
        # Extract entities and relations from original document with concurrency
        logger.info(f"Extracting entities and relations from {source_language} document")
        
        # We can run the entity extraction and getting the entities/relations in parallel
        extraction_tasks = [
            extract_entities(
                original_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                llm_response_cache=self.llm_response_cache,
                global_config=asdict(self),
            )
        ]
        
        # Wait for extraction to complete
        await asyncio.gather(*extraction_tasks)
        
        # Get entities and relations from original document
        doc_data_tasks = [
            self._get_document_entities(original_doc_id, original_chunks),
            self._get_document_relations(original_doc_id, original_chunks)
        ]
        
        original_entities, original_relations = await asyncio.gather(*doc_data_tasks)
        
        logger.info(f"Found {len(original_entities)} entities and {len(original_relations)} relations in {source_language} document")
        
        # Update status for original document
        await self.doc_status.upsert({
            original_doc_id: {
                "status": DocStatus.PROCESSED,
                "chunks_count": len(original_chunks),
                "content": data_original,
                "content_summary": self._get_content_summary(data_original),
                "content_length": len(data_original),
                "updated_at": datetime.now().isoformat(),
            }
        })
        
        # Extract corresponding entities and relations in the translated document using parallel processing
        logger.info(f"Extracting matching entities and relations from {target_language} document")
        
        # Extract entities and relations in parallel
        matching_tasks = [
            self._extract_matching_entities(
                original_entities,
                data_translated,
                source_language,
                target_language,
                translated_chunks
            ),
            # We'll get relations after entities are extracted
        ]
        
        translated_entities = await matching_tasks[0]
        
        # Now we can get the relations using the translated entities
        translated_relations = await self._extract_matching_relations(
            original_relations,
            translated_entities,
            data_translated,
            source_language,
            target_language,
            translated_chunks
        )
        
        logger.info(f"Extracted {len(translated_entities)} entities and {len(translated_relations)} relations in {target_language} document")
        
        # Verify counts match and fix if needed - can run in parallel
        verification_tasks = []
        
        if len(original_entities) != len(translated_entities):
            logger.warning(f"Entity count mismatch: {len(original_entities)} {source_language} vs {len(translated_entities)} {target_language}")
            # Force entity count to match by requesting a fix
            verification_tasks.append(
                self._fix_entity_count_mismatch(
                    original_entities,
                    translated_entities,
                    data_translated,
                    source_language,
                    target_language,
                    translated_chunks
                )
            )
        
        if len(original_relations) != len(translated_relations):
            logger.warning(f"Relation count mismatch: {len(original_relations)} {source_language} vs {len(translated_relations)} {target_language}")
            # Force relation count to match by requesting a fix
            verification_tasks.append(
                self._fix_relation_count_mismatch(
                    original_relations,
                    translated_relations, 
                    translated_entities,
                    data_translated,
                    source_language, 
                    target_language,
                    translated_chunks
                )
            )
        
        # Execute verification tasks if needed
        if verification_tasks:
            verification_results = await asyncio.gather(*verification_tasks)
            
            # Update entities and relations with fixed versions
            if len(original_entities) != len(translated_entities):
                translated_entities = verification_results[0]
                verification_results = verification_results[1:]
            
            if len(original_relations) != len(translated_relations):
                translated_relations = verification_results[0]
        
        # Prepare data for batch operations - this is for translated entities/relations
        nodes_data_map = {}
        for entity in translated_entities:
            entity_name = f'"{entity["name"].upper()}"'
            
            # Get first chunk ID for this document
            chunk_id = next(iter(translated_chunks.keys()))
            
            # Prepare data
            if entity_name not in nodes_data_map:
                nodes_data_map[entity_name] = []
            print("Extracted entity: ", entity["type"].upper())
            if entity["type"].lower() not in PROMPTS["DEFAULT_ENTITY_TYPES"] + PROMPTS["DEFAULT_ENTITY_TYPES_VI"]:
                entity["type"] = "UNKNOWN"
            nodes_data_map[entity_name].append({
                "entity_type": f'"{entity["type"].upper()}"',
                "description": entity["description"],
                "source_id": chunk_id,
                "language": target_language,
            })

        # Prepare data for edges
        edges_data_map = {}
        for relation in translated_relations:
            src_entity = f'"{relation["source"].upper()}"'
            tgt_entity = f'"{relation["target"].upper()}"'
            
            # Get first chunk ID for this document
            chunk_id = next(iter(translated_chunks.keys()))
            
            edge_key = (src_entity, tgt_entity)
            if edge_key not in edges_data_map:
                edges_data_map[edge_key] = []
            
            edges_data_map[edge_key].append({
                "description": relation["description"],
                "keywords": relation["keywords"],
                "weight": 1.0,
                "source_id": chunk_id,
                "language": target_language,
            })
        
        # Create tasks for entity and relation merging
        merge_tasks = []
        
        # Add entity merge tasks
        for entity_name, nodes_data in nodes_data_map.items():
            merge_tasks.append(
                _merge_nodes_then_upsert(
                    entity_name, 
                    nodes_data,
                    self.chunk_entity_relation_graph, 
                    asdict(self)
                )
            )
        
        # Add relation merge tasks
        for (src_id, tgt_id), edges_data in edges_data_map.items():
            merge_tasks.append(
                _merge_edges_then_upsert(
                    src_id, 
                    tgt_id, 
                    edges_data,
                    self.chunk_entity_relation_graph, 
                    asdict(self)
                )
            )
        
        # Execute all merge operations in parallel
        all_merge_results = await asyncio.gather(*merge_tasks)
        
        # Split results into entities and relationships
        # First n results are entities, where n is the number of entities
        entity_count = len(nodes_data_map)
        all_entities_data = all_merge_results[:entity_count]
        all_relationships_data = all_merge_results[entity_count:]
        
        # Prepare data for vector databases - do this in separate loop to not block merge operations
        entities_vdb_data = {}
        for entity_data in all_entities_data:
            if not entity_data:
                continue
            
            entity_id = compute_mdhash_id(entity_data["entity_name"], prefix="ent-")
            entities_vdb_data[entity_id] = {
                "content": f"{entity_data['entity_name']} {entity_data['description']}",
                "entity_name": entity_data["entity_name"],
                "language": entity_data.get("language", target_language),
            }

        relationships_vdb_data = {}
        for rel_data in all_relationships_data:
            if not rel_data:
                continue
                
            relation_id = compute_mdhash_id(rel_data["src_id"] + rel_data["tgt_id"], prefix="rel-")
            relationships_vdb_data[relation_id] = {
                "content": f"{rel_data['keywords']} {rel_data['src_id']} {rel_data['tgt_id']} {rel_data['description']}",
                "src_id": rel_data["src_id"],
                "tgt_id": rel_data["tgt_id"],
                "language": rel_data.get("language", target_language),
            }
        
        # Final tasks to run in parallel
        final_tasks = [
            # Update vector databases
            self.entities_vdb.upsert(entities_vdb_data),
            self.relationships_vdb.upsert(relationships_vdb_data),
            
            # Update document status
            self.doc_status.upsert({
                translated_doc_id: {
                    "status": DocStatus.PROCESSED,
                    "chunks_count": len(translated_chunks),
                    "content": data_translated,
                    "content_summary": self._get_content_summary(data_translated),
                    "content_length": len(data_translated),
                    "updated_at": datetime.now().isoformat(),
                }
            }),
            
            # Create cross-lingual edges
            self._create_cross_lingual_edges(
                original_entities,
                translated_entities,
                source_language,
                target_language
            )
        ]
        
        # Add translation storage if requested
        if store_translations:
            final_tasks.append(
                self._store_translation_pairs(
                    original_entities,
                    translated_entities,
                    original_relations,
                    translated_relations,
                    source_language,
                    target_language,
                    translation_db_path
                )
            )
        
        # Execute all final tasks in parallel
        await asyncio.gather(*final_tasks)
        
        # Save changes to all storages
        await self._insert_done()
        
        logger.info(f"Duo insertion completed successfully")
        return original_doc_id, translated_doc_id

    async def _get_document_entities(self, doc_id: str, chunks: dict = None) -> list[dict]:
        """
        Get entities associated with a specific document.
        
        Args:
            doc_id: Document ID
            chunks: Optional dictionary of chunks to use for matching
            
        Returns:
            List of entity dictionaries with name, type, and description
        """
        # Get all chunks for this document if not provided
        if chunks is None:
            doc_chunks = {}
            chunks_data = await self.text_chunks.get_all()
            for chunk_id, chunk_data in chunks_data.items():
                if chunk_data.get('full_doc_id') == doc_id:
                    doc_chunks[chunk_id] = chunk_data
        else:
            doc_chunks = chunks
        
        if not doc_chunks:
            logger.warning(f"No chunks found for document {doc_id}")
            return []
        
        # Get entities that have these chunks as source
        all_nodes = self.chunk_entity_relation_graph._graph.nodes(data=True)
        doc_entities = []
        
        for node_id, data in all_nodes:
            if 'source_id' in data:
                # Check if any source is from the document's chunks
                sources = data['source_id'].split(GRAPH_FIELD_SEP)
                for chunk_id in doc_chunks:
                    if chunk_id in sources:
                        # Add entity with its metadata
                        entity_data = {
                            "name": node_id.strip('"'),
                            "type": data.get('entity_type', '').strip('"'),
                            "description": data.get('description', '')
                        }
                        doc_entities.append(entity_data)
                        break
        
        return doc_entities

    async def _get_document_relations(self, doc_id: str, chunks: dict = None) -> list[dict]:
        """
        Get relations associated with a specific document.
        
        Args:
            doc_id: Document ID
            chunks: Optional dictionary of chunks to use for matching
            
        Returns:
            List of relation dictionaries with source, target, and description
        """
        # Get all chunks for this document if not provided
        if chunks is None:
            doc_chunks = {}
            chunks_data = await self.text_chunks.get_all()
            for chunk_id, chunk_data in chunks_data.items():
                if chunk_data.get('full_doc_id') == doc_id:
                    doc_chunks[chunk_id] = chunk_data
        else:
            doc_chunks = chunks
        
        if not doc_chunks:
            logger.warning(f"No chunks found for document {doc_id}")
            return []
        
        # Get relations that have these chunks as source
        all_edges = self.chunk_entity_relation_graph._graph.edges(data=True)
        doc_relations = []
        
        for src, tgt, data in all_edges:
            if 'source_id' in data:
                # Check if any source is from the document's chunks
                sources = data['source_id'].split(GRAPH_FIELD_SEP)
                for chunk_id in doc_chunks:
                    if chunk_id in sources:
                        # Add relation with its metadata
                        relation_data = {
                            "source": src.strip('"'),
                            "target": tgt.strip('"'),
                            "description": data.get('description', ''),
                            "keywords": data.get('keywords', '')
                        }
                        doc_relations.append(relation_data)
                        break
        
        return doc_relations

    async def _translate_preserving_structure(
        self, content: str, source_language: str, target_language: str
    ) -> str:
        """
        Translate content while preserving semantic structure.
        
        Args:
            content: Content to translate
            source_language: Source language
            target_language: Target language
            
        Returns:
            Translated content
        """
        prompt = f"""
        You are a professional translator specialized in {source_language}-to-{target_language} translation.
        
        Please translate the following {source_language} text into {target_language}.
        Your translation must:
        1. Maintain the exact same meaning and information as the original
        2. Preserve all named entities (people, organizations, locations, etc.)
        3. Keep the same document structure and flow
        4. Sound natural in {target_language}
        
        Original {source_language} text:
        {content}
        
        {target_language} translation:
        """
        
        response = await self.llm_model_func(prompt)
        return response.strip()

    async def _extract_matching_entities(
        self, 
        original_entities: list[dict],
        translated_text: str,
        source_language: str,
        target_language: str,
        translated_chunks: dict
    ) -> list[dict]:
        """
        Extract entities from translated text that correspond to original entities.
        This ensures a 1:1 mapping between source and target entities.
        
        Args:
            original_entities: List of original entities
            translated_text: Translated document text
            source_language: Source language
            target_language: Target language
            translated_chunks: Dictionary of translated chunks
            
        Returns:
            List of entity dictionaries in the target language
        """
        # Format original entities list for the prompt
        formatted_entities = []
        for i, entity in enumerate(original_entities):
            formatted_entities.append(f"{i+1}. {entity['name']} (Type: {entity['type']}): {entity['description']}")
        
        original_entities_str = "\n".join(formatted_entities)
        
        # Create prompt for entity extraction
        prompt = f"""
        You are tasked with finding the exact equivalent entities in a {target_language} text 
        that correspond to entities identified in the original {source_language} text.
        
        I have identified the following entities in the {source_language} text:
        {original_entities_str}
        
        Now I need you to identify the EXACT corresponding entities in this {target_language} text:
        {translated_text}
        
        IMPORTANT REQUIREMENTS:
        1. You MUST identify EXACTLY {len(original_entities)} entities in the {target_language} text.
        2. Each entity must correspond to one entity in the {source_language} list in the same order.
        3. The entity types should remain the same.
        4. If an entity name is a proper noun or name, it may be spelled the same in both languages.
        5. IMPORTANT: You MUST provide the description in {target_language}, not in {source_language}. Even if you see the entity in the text with a description in {source_language}, you need to translate that description to {target_language}.
        
        Format your response EXACTLY as follows, with one entity per line:
        1. Entity: [entity name] | Type: [entity type] | Description: [description in {target_language}]
        2. Entity: [entity name] | Type: [entity type] | Description: [description in {target_language}]
        ...and so on until you have EXACTLY {len(original_entities)} entities.
        
        Think carefully about each entity and ensure you've found its most accurate equivalent.
        Make sure all descriptions are in {target_language}, not in {source_language}.
        """
        
        response = await self.llm_model_func(prompt)
        
        # Parse the response to extract entity information
        translated_entities = []
        entity_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '|' in line]
        
        for line in entity_lines:
            try:
                # Remove any numbering at the start
                line = re.sub(r'^\d+\.\s*', '', line)
                
                # Extract entity details
                entity_parts = line.split('|')
                if len(entity_parts) >= 3:
                    entity_name = entity_parts[0].split('Entity:')[1].strip() if 'Entity:' in entity_parts[0] else entity_parts[0].strip()
                    entity_type = entity_parts[1].split('Type:')[1].strip() if 'Type:' in entity_parts[1] else entity_parts[1].strip()
                    entity_desc = entity_parts[2].split('Description:')[1].strip() if 'Description:' in entity_parts[2] else entity_parts[2].strip()
                    if entity_type.lower() not in PROMPTS["DEFAULT_ENTITY_TYPES"] + PROMPTS["DEFAULT_ENTITY_TYPES_VI"]:
                        entity_type = "UNKNOWN"
                    translated_entities.append({
                        "name": entity_name,
                        "type": entity_type,
                        "description": entity_desc
                    })
            except Exception as e:
                logger.warning(f"Error parsing entity line: {line}. Error: {e}")
        
        # Only return the extracted entities, don't save to graph yet
        return translated_entities

    async def _extract_matching_relations(
        self,
        original_relations: list[dict],
        translated_entities: list[dict],
        translated_text: str,
        source_language: str,
        target_language: str,
        translated_chunks: dict
    ) -> list[dict]:
        """
        Extract relations from translated text that correspond to original relations.
        
        Args:
            original_relations: List of original relations
            translated_entities: List of translated entities
            translated_text: Translated document text
            source_language: Source language
            target_language: Target language
            translated_chunks: Dictionary of translated chunks
            
        Returns:
            List of relation dictionaries in the target language
        """
        if not original_relations:
            return []
        
        # Create entity name mapping for easy lookup
        entity_name_map = {entity["name"].upper(): entity["name"] for entity in translated_entities}
        
        # Format original relations list for the prompt
        formatted_relations = []
        for i, relation in enumerate(original_relations):
            formatted_relations.append(f"{i+1}. {relation['source']} → {relation['target']}: {relation['description']}")
        
        original_relations_str = "\n".join(formatted_relations)
        
        # Format translated entities for reference
        formatted_entities = []
        for i, entity in enumerate(translated_entities):
            formatted_entities.append(f"{i+1}. {entity['name']} (Type: {entity['type']})")
        
        translated_entities_str = "\n".join(formatted_entities)
        
        # Create prompt for relation extraction
        prompt = f"""
        You are tasked with finding the exact equivalent relationships in a {target_language} text 
        that correspond to relationships identified in the original {source_language} text.
        
        I have identified the following relationships in the {source_language} text:
        {original_relations_str}
        
        I have already identified these entities in the {target_language} text:
        {translated_entities_str}
        
        Now I need you to identify the EXACT corresponding relationships in this {target_language} text:
        {translated_text}
        
        IMPORTANT REQUIREMENTS:
        1. You MUST identify EXACTLY {len(original_relations)} relationships in the {target_language} text.
        2. Each relationship must correspond to one relationship in the {source_language} list in the same order.
        3. ONLY use entities from the list of {target_language} entities I provided.
        4. The relationships should maintain the same directional meaning (A → B).
        5. IMPORTANT: You MUST provide the description in {target_language}, not in {source_language}. Even if you see the relationship in the text with a description in {source_language}, you need to translate that description to {target_language}.
        
        Format your response EXACTLY as follows, with one relationship per line:
        1. [source entity] → [target entity] | Description: [relationship description in {target_language}] | Keywords: [comma-separated keywords in {target_language}]
        2. [source entity] → [target entity] | Description: [relationship description in {target_language}] | Keywords: [comma-separated keywords in {target_language}]
        ...and so on until you have EXACTLY {len(original_relations)} relationships.
        
        Think carefully about each relationship and ensure you've found its most accurate equivalent.
        Make sure all descriptions and keywords are in {target_language}, not in {source_language}.
        """
        
        response = await self.llm_model_func(prompt)
        
        # Parse the response to extract relation information
        translated_relations = []
        relation_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '→' in line]
        
        for line in relation_lines:
            try:
                # Remove any numbering at the start
                line = re.sub(r'^\d+\.\s*', '', line)
                
                # Split into relationship and metadata
                rel_parts = line.split('|')
                if len(rel_parts) >= 2:
                    # Parse the relationship part (source → target)
                    rel_entities = rel_parts[0].strip().split('→')
                    if len(rel_entities) == 2:
                        src_entity = rel_entities[0].strip()
                        tgt_entity = rel_entities[1].strip()
                        
                        # Parse description and keywords
                        rel_desc = ""
                        rel_keywords = ""
                        
                        for part in rel_parts[1:]:
                            if 'Description:' in part:
                                rel_desc = part.split('Description:')[1].strip()
                            elif 'Keywords:' in part:
                                rel_keywords = part.split('Keywords:')[1].strip()
                        
                        translated_relations.append({
                            "source": src_entity,
                            "target": tgt_entity,
                            "description": rel_desc,
                            "keywords": rel_keywords
                        })
            except Exception as e:
                logger.warning(f"Error parsing relationship line: {line}. Error: {e}")
        
        # Only return the extracted relations, don't save to graph yet
        return translated_relations

    async def _store_translation_pairs(
        self, 
        original_entities: list[dict], 
        translated_entities: list[dict],
        original_relations: list[dict], 
        translated_relations: list[dict],
        source_language: str, 
        target_language: str, 
        db_path: str
    ) -> None:
        """
        Store entity and relation translation pairs in a JSON file for future use.
        
        Args:
            original_entities: Entities from original text
            translated_entities: Entities from translated text
            original_relations: Relations from original text
            translated_relations: Relations from translated text
            source_language: Source language name
            target_language: Target language name
            db_path: Path to save the translation database
        """
        from lightrag.utils import load_json, write_json
        
        # Ensure we have equal numbers of entities and relations
        if len(original_entities) != len(translated_entities):
            # Take the minimum number to ensure pairs
            min_count = min(len(original_entities), len(translated_entities))
            original_entities = original_entities[:min_count]
            translated_entities = translated_entities[:min_count]
            
        if len(original_relations) != len(translated_relations):
            # Take the minimum number to ensure pairs
            min_count = min(len(original_relations), len(translated_relations))
            original_relations = original_relations[:min_count]
            translated_relations = translated_relations[:min_count]
        
        # Create entity translation pairs
        entity_translations = []
        for i in range(len(original_entities)):
            entity_translations.append({
                "original": original_entities[i]["name"],
                "translated": translated_entities[i]["name"],
                "original_description": original_entities[i]["description"],
                "translated_description": translated_entities[i]["description"],
                "original_language": source_language,
                "translated_language": target_language,
                "type": original_entities[i]["type"],
                "timestamp": datetime.now().isoformat()
            })
        
        # Create relation translation pairs
        relation_translations = []
        for i in range(len(original_relations)):
            relation_translations.append({
                "original_src": original_relations[i]["source"],
                "original_tgt": original_relations[i]["target"],
                "original_desc": original_relations[i]["description"],
                "translated_src": translated_relations[i]["source"],
                "translated_tgt": translated_relations[i]["target"],
                "translated_desc": translated_relations[i]["description"],
                "original_language": source_language,
                "translated_language": target_language,
                "timestamp": datetime.now().isoformat()
            })
        
        # Load existing translations if file exists
        translations_db = {}
        try:
            if os.path.exists(db_path):
                translations_db = load_json(db_path) or {}
        except Exception as e:
            logger.warning(f"Could not load existing translations: {e}")
        
        # Add new translations
        if 'entities' not in translations_db:
            translations_db['entities'] = []
        if 'relations' not in translations_db:
            translations_db['relations'] = []
        
        translations_db['entities'].extend(entity_translations)
        translations_db['relations'].extend(relation_translations)
        
        # Remove duplicates
        translations_db['entities'] = self._deduplicate_translations(translations_db['entities'])
        translations_db['relations'] = self._deduplicate_translations(translations_db['relations'])
        
        # Save to file
        write_json(translations_db, db_path)
        logger.info(f"Saved {len(entity_translations)} entity and {len(relation_translations)} relation translations to {db_path}")

    def _deduplicate_translations(self, translations_list: list[dict]) -> list[dict]:
        """
        Remove duplicate translations from a list of translation pairs.
        
        Args:
            translations_list: List of translation dictionaries
            
        Returns:
            Deduplicated list of translation dictionaries
        """
        seen = set()
        unique_translations = []
        
        for trans in translations_list:
            # Use relevant fields to create a unique key
            if 'original' in trans and 'translated' in trans:
                key = f"{trans['original']}|{trans['translated']}|{trans.get('original_language', '')}|{trans.get('translated_language', '')}"
            elif 'original_src' in trans and 'original_tgt' in trans:
                key = (f"{trans['original_src']}|{trans['original_tgt']}|"
                    f"{trans['translated_src']}|{trans['translated_tgt']}|"
                    f"{trans.get('original_language', '')}|{trans.get('translated_language', '')}")
            else:
                # If we can't determine a unique key, keep the entry
                unique_translations.append(trans)
                continue
            
            if key not in seen:
                seen.add(key)
                unique_translations.append(trans)
        
        return unique_translations
    async def _fix_entity_count_mismatch(
        self, 
        original_entities: list[dict],
        translated_entities: list[dict],
        translated_text: str,
        source_language: str,
        target_language: str,
        translated_chunks: dict
    ) -> list[dict]:
        """
        Fix entity count mismatch by requesting additional entities or removing excess ones.
        
        Args:
            original_entities: List of original entities
            translated_entities: List of translated entities (potentially mismatched)
            translated_text: Translated document text
            source_language: Source language
            target_language: Target language
            translated_chunks: Dictionary of translated chunks
            
        Returns:
            Updated list of translated entities with correct count
        """
        orig_count = len(original_entities)
        trans_count = len(translated_entities)
        
        if orig_count == trans_count:
            return translated_entities
        
        # Format original entities
        orig_entities_str = "\n".join([
            f"{i+1}. {entity['name']} (Type: {entity['type']}): {entity['description']}"
            for i, entity in enumerate(original_entities)
        ])
        
        # Format existing translated entities
        trans_entities_str = "\n".join([
            f"{i+1}. {entity['name']} (Type: {entity['type']}): {entity['description']}"
            for i, entity in enumerate(translated_entities)
        ])
        
        # Build the appropriate prompt based on the mismatch type
        if trans_count < orig_count:
            # We need more entities
            prompt = f"""
            I need your help fixing an entity count mismatch between {source_language} and {target_language} versions of a document.
            
            The {source_language} document has {orig_count} entities:
            {orig_entities_str}
            
            But we only identified {trans_count} entities in the {target_language} document:
            {trans_entities_str}
            
            Please identify {orig_count - trans_count} ADDITIONAL entities in the {target_language} text that would complete the list,
            ensuring we have exactly one {target_language} entity for each {source_language} entity.
            
            {target_language} text:
            {translated_text}
            
            Format your response EXACTLY as follows for ONLY the ADDITIONAL entities:
            {trans_count+1}. Entity: [entity name] | Type: [entity type] | Description: [description]
            {trans_count+2}. Entity: [entity name] | Type: [entity type] | Description: [description]
            ...and so on until we have a total of {orig_count} entities.
            """
            
            response = await self.llm_model_func(prompt)
            
            # Parse the additional entities
            additional_entities = []
            entity_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '|' in line]
            
            for line in entity_lines:
                try:
                    # Remove any numbering at the start
                    line = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Extract entity details
                    entity_parts = line.split('|')
                    if len(entity_parts) >= 3:
                        entity_name = entity_parts[0].split('Entity:')[1].strip() if 'Entity:' in entity_parts[0] else entity_parts[0].strip()
                        entity_type = entity_parts[1].split('Type:')[1].strip() if 'Type:' in entity_parts[1] else entity_parts[1].strip()
                        entity_desc = entity_parts[2].split('Description:')[1].strip() if 'Description:' in entity_parts[2] else entity_parts[2].strip()
                        
                        additional_entities.append({
                            "name": entity_name,
                            "type": entity_type,
                            "description": entity_desc
                        })
                except Exception as e:
                    logger.warning(f"Error parsing entity line: {line}. Error: {e}")
            
            # Combine existing and additional entities
            updated_entities = translated_entities + additional_entities
            
        else:
            # We have too many entities - keep only the most important ones
            prompt = f"""
            I need your help fixing an entity count mismatch between {source_language} and {target_language} versions of a document.
            
            The {source_language} document has {orig_count} entities:
            {orig_entities_str}
            
            But we identified {trans_count} entities in the {target_language} document:
            {trans_entities_str}
            
            Please select EXACTLY {orig_count} entities from the {target_language} list that best match the entities in the {source_language} list.
            
            Format your response EXACTLY as follows for the SELECTED entities:
            1. Entity: [entity name] | Type: [entity type] | Description: [description]
            2. Entity: [entity name] | Type: [entity type] | Description: [description]
            ...and so on until you have EXACTLY {orig_count} entities.
            
            Make sure that each selected entity corresponds with one of the original entities in order.
            """
            
            response = await self.llm_model_func(prompt)
            
            # Parse the selected entities
            updated_entities = []
            entity_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '|' in line]
            
            for line in entity_lines:
                try:
                    # Remove any numbering at the start
                    line = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Extract entity details
                    entity_parts = line.split('|')
                    if len(entity_parts) >= 3:
                        entity_name = entity_parts[0].split('Entity:')[1].strip() if 'Entity:' in entity_parts[0] else entity_parts[0].strip()
                        entity_type = entity_parts[1].split('Type:')[1].strip() if 'Type:' in entity_parts[1] else entity_parts[1].strip()
                        entity_desc = entity_parts[2].split('Description:')[1].strip() if 'Description:' in entity_parts[2] else entity_parts[2].strip()
                        
                        updated_entities.append({
                            "name": entity_name,
                            "type": entity_type,
                            "description": entity_desc
                        })
                except Exception as e:
                    logger.warning(f"Error parsing entity line: {line}. Error: {e}")
        
        # Verify the count is correct now
        if len(updated_entities) != orig_count:
            logger.warning(f"Entity count still mismatched after fixing: {len(updated_entities)} vs {orig_count}")
            # Force the count to match by truncating or padding
            if len(updated_entities) > orig_count:
                updated_entities = updated_entities[:orig_count]
            else:
                # Duplicate the last entity if needed
                while len(updated_entities) < orig_count:
                    duplicate = dict(updated_entities[-1])
                    duplicate["name"] = f"{duplicate['name']} (Copy)"
                    updated_entities.append(duplicate)
        
        return updated_entities

    async def _fix_relation_count_mismatch(
        self,
        original_relations: list[dict],
        translated_relations: list[dict], 
        translated_entities: list[dict],
        data_translated: str,
        source_language: str, 
        target_language: str,
        translated_chunks: dict
    ) -> list[dict]:
        """
        Fix relation count mismatch by requesting additional relations or removing excess ones.
        
        Args:
            original_relations: List of original relations
            translated_relations: List of translated relations (potentially mismatched)
            translated_entities: List of translated entities
            data_translated: Translated document text
            source_language: Source language
            target_language: Target language
            translated_chunks: Dictionary of translated chunks
            
        Returns:
            Updated list of translated relations with correct count
        """
        orig_count = len(original_relations)
        trans_count = len(translated_relations)
        
        if orig_count == trans_count:
            return translated_relations
        
        # Format original relations
        orig_relations_str = "\n".join([
            f"{i+1}. {rel['source']} → {rel['target']}: {rel['description']}"
            for i, rel in enumerate(original_relations)
        ])
        
        # Format existing translated relations
        trans_relations_str = "\n".join([
            f"{i+1}. {rel['source']} → {rel['target']}: {rel['description']}"
            for i, rel in enumerate(translated_relations)
        ])
        
        # Format translated entities for reference
        trans_entities_str = "\n".join([
            f"{i+1}. {entity['name']} (Type: {entity['type']})"
            for i, entity in enumerate(translated_entities)
        ])
        
        # Build the appropriate prompt based on the mismatch type
        if trans_count < orig_count:
            # We need more relations
            prompt = f"""
            I need your help fixing a relationship count mismatch between {source_language} and {target_language} versions of a document.
            
            The {source_language} document has {orig_count} relationships:
            {orig_relations_str}
            
            But we only identified {trans_count} relationships in the {target_language} document:
            {trans_relations_str}
            
            Here are the entities available in the {target_language} document:
            {trans_entities_str}
            
            Please identify {orig_count - trans_count} ADDITIONAL relationships in the {target_language} text that would complete the list,
            ensuring we have exactly one {target_language} relationship for each {source_language} relationship.
            
            {target_language} text:
            {data_translated}
            
            Format your response EXACTLY as follows for ONLY the ADDITIONAL relationships:
            {trans_count+1}. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
            {trans_count+2}. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
            ...and so on until we have a total of {orig_count} relationships.
            
            ONLY use entity names from the list I provided above.
            """
            
            response = await self.llm_model_func(prompt)
            
            # Parse the additional relations
            additional_relations = []
            relation_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '→' in line]
            
            for line in relation_lines:
                try:
                    # Remove any numbering at the start
                    line = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Split into relationship and metadata
                    rel_parts = line.split('|')
                    if len(rel_parts) >= 2:
                        # Parse the relationship part (source → target)
                        rel_entities = rel_parts[0].strip().split('→')
                        if len(rel_entities) == 2:
                            src_entity = rel_entities[0].strip()
                            tgt_entity = rel_entities[1].strip()
                            
                            # Parse description and keywords
                            rel_desc = ""
                            rel_keywords = ""
                            
                            for part in rel_parts[1:]:
                                if 'Description:' in part:
                                    rel_desc = part.split('Description:')[1].strip()
                                elif 'Keywords:' in part:
                                    rel_keywords = part.split('Keywords:')[1].strip()
                            
                            additional_relations.append({
                                "source": src_entity,
                                "target": tgt_entity,
                                "description": rel_desc,
                                "keywords": rel_keywords
                            })
                except Exception as e:
                    logger.warning(f"Error parsing relationship line: {line}. Error: {e}")
            
            # Combine existing and additional relations
            updated_relations = translated_relations + additional_relations
            
        else:
            # We have too many relations - keep only the most important ones
            prompt = f"""
            I need your help fixing a relationship count mismatch between {source_language} and {target_language} versions of a document.
            
            The {source_language} document has {orig_count} relationships:
            {orig_relations_str}
            
            But we identified {trans_count} relationships in the {target_language} document:
            {trans_relations_str}
            
            Please select EXACTLY {orig_count} relationships from the {target_language} list that best match the relationships in the {source_language} list.
            
            Format your response EXACTLY as follows for the SELECTED relationships:
            1. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
            2. [source entity] → [target entity] | Description: [relationship description] | Keywords: [comma-separated keywords]
            ...and so on until you have EXACTLY {orig_count} relationships.
            
            Make sure that each selected relationship corresponds with one of the original relationships in order.
            """
            
            response = await self.llm_model_func(prompt)
            
            # Parse the selected relations
            updated_relations = []
            relation_lines = [line.strip() for line in response.strip().split('\n') if line.strip() and '→' in line]
            
            for line in relation_lines:
                try:
                    # Remove any numbering at the start
                    line = re.sub(r'^\d+\.\s*', '', line)
                    
                    # Split into relationship and metadata
                    rel_parts = line.split('|')
                    if len(rel_parts) >= 2:
                        # Parse the relationship part (source → target)
                        rel_entities = rel_parts[0].strip().split('→')
                        if len(rel_entities) == 2:
                            src_entity = rel_entities[0].strip()
                            tgt_entity = rel_entities[1].strip()
                            
                            # Parse description and keywords
                            rel_desc = ""
                            rel_keywords = ""
                            
                            for part in rel_parts[1:]:
                                if 'Description:' in part:
                                    rel_desc = part.split('Description:')[1].strip()
                                elif 'Keywords:' in part:
                                    rel_keywords = part.split('Keywords:')[1].strip()
                            
                            updated_relations.append({
                                "source": src_entity,
                                "target": tgt_entity,
                                "description": rel_desc,
                                "keywords": rel_keywords
                            })
                except Exception as e:
                    logger.warning(f"Error parsing relationship line: {line}. Error: {e}")
        
        # Verify the count is correct now
        if len(updated_relations) != orig_count:
            logger.warning(f"Relation count still mismatched after fixing: {len(updated_relations)} vs {orig_count}")
            # Force the count to match by truncating or padding
            if len(updated_relations) > orig_count:
                updated_relations = updated_relations[:orig_count]
            else:
                # Duplicate the last relation if needed
                while len(updated_relations) < orig_count:
                    duplicate = dict(updated_relations[-1])
                    duplicate["description"] = f"{duplicate['description']} (Copy)"
                    updated_relations.append(duplicate)
        
        return updated_relations

    async def _create_cross_lingual_edges(
        self, 
        original_entities: list[dict], 
        translated_entities: list[dict],
        source_language: str, 
        target_language: str
    ) -> None:
        """
        Create edges in the knowledge graph to connect original entities with their translations.
        
        Args:
            original_entities: List of entities from original text
            translated_entities: List of entities from translated text
            source_language: Source language name
            target_language: Target language name
        """
        # Ensure we have equal numbers of entities
        if len(original_entities) != len(translated_entities):
            logger.warning(f"Entity count mismatch: {len(original_entities)} {source_language} vs {len(translated_entities)} {target_language}")
            # Take the minimum number to ensure pairs
            min_count = min(len(original_entities), len(translated_entities))
            original_entities = original_entities[:min_count]
            translated_entities = translated_entities[:min_count]
        
        # Create cross-lingual edges - one edge for each entity pair
        for i in range(len(original_entities)):
            orig_entity = original_entities[i]
            trans_entity = translated_entities[i]
            
            orig_name = f'"{orig_entity["name"].upper()}"'
            trans_name = f'"{trans_entity["name"].upper()}"'
            
            # Create the cross-lingual edge
            edge_data = {
                "weight": 1.0,
                "description": f"Translation equivalent ({source_language} ↔ {target_language})",
                "keywords": f"translation,cross-lingual,{source_language.lower()},{target_language.lower()}",
                "relation_type": "translation_equivalent",
                "languages": f"{source_language},{target_language}",
                "source_id": "cross_lingual",
                "original_language": source_language,
                "translated_language": target_language,
                "created_at": datetime.now().isoformat(),
                "confidence_score": 1.0,
            }
            
            # Add to knowledge graph
            await self.chunk_entity_relation_graph.upsert_edge(orig_name, trans_name, edge_data)
            
            logger.debug(f"Created cross-lingual edge between {orig_name} ({source_language}) and {trans_name} ({target_language})")
            
        logger.info(f"Created {len(original_entities)} cross-lingual edges between {source_language} and {target_language} entities")

    def create_db_with_new_embedding(self, embedding_func, embedding_name):
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.a_create_db_with_new_embedding(embedding_func, embedding_name)
        )

    async def a_create_db_with_new_embedding(self,
                                     embedding_func,
                                     embedding_name):        
        # create db for entities, relationships and chunks:
        entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES + "_" + embedding_name
            ),
            embedding_func=embedding_func,
            meta_fields={"entity_name", "chunk_id", "description"},
        )

        relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS + "_" + embedding_name
            ),
            embedding_func=embedding_func,
            meta_fields={"src_id", "tgt_id", "chunk_id", "description"},
        )

        chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS + "_" + embedding_name
            ),
            embedding_func=embedding_func,
        )

        # Intialize the database:
        tasks = []

        for storage in (
                entities_vdb,
                relationships_vdb,
                chunks_vdb,
            ):
                if storage:
                    tasks.append(storage.initialize())

        await asyncio.gather(*tasks)

        # Take the data from entities, relationships and chunks to upsert
        nodes = list(self.chunk_entity_relation_graph._graph.nodes(data = True))
        all_entities_data = []
        for entity_name, node_data in nodes:
            new_node_data = {"entity_name" : entity_name, **node_data}
            all_entities_data.append(new_node_data)

        # data_for_vdb = {
        #     compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
        #         "content": dp["entity_name"] + dp["description"],
        #         "entity_name": dp["entity_name"],
        #     }
        #     for dp in all_entities_data
        # }

        # data_for_vdb = {
        #     compute_mdhash_id(dp["entity_name"] + dp["description_list"], prefix="ent-"): {
        #         "content": dp["entity_name"],
        #         "entity_name": dp["entity_name"],
        #     }
        #     for dp in all_entities_data
        # }
        print("Embedding Nodes ...")
        print("Add_dess")
        data_for_vdb = {}
        for dp in all_entities_data:
            list_des = list(dp["description"].split("<SEP>"))
            list_src = list(dp["source_id"].split("<SEP>"))
            for des_id in range(len(list_des)):
                des = list_des[des_id]
                chunk_id = list_src[des_id]
                data_for_vdb[compute_mdhash_id(dp["entity_name"] + des, prefix="ent-")] = {
                    "content": dp["entity_name"] + " " + des,
                    "entity_name": dp["entity_name"],
                    "chunk_id" : chunk_id,
                    "description" : des
                }
        # print("Embedding Nodes ...")
        # data_for_vdb = {}
        # for dp in all_entities_data:
        #     list_des = list(dp["description_list"].split("<SEP>"))
        #     for des_id in range(len(list_des)):
        #         des = list_des[des_id]
        #         data_for_vdb[compute_mdhash_id(dp["entity_name"] + des, prefix="ent-")] = {
        #             "content": dp["entity_name"] + " " + des,
        #             "entity_name": dp["entity_name"],
        #         }
        await entities_vdb.upsert(data_for_vdb)

        ## Take the edges data:
        edges = list(self.chunk_entity_relation_graph._graph.edges(data = True))
        all_relationships_data = []
        for src_id, tgt_id, edge_data in edges:
            new_edge_data = {"src_id" : src_id, "tgt_id" : tgt_id, **edge_data}
            all_relationships_data.append(new_edge_data)
        # data_for_vdb = {
        #     compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
        #         "src_id": dp["src_id"],
        #         "tgt_id": dp["tgt_id"],
        #         "content": dp["keywords"]
        #         + dp["src_id"]
        #         + dp["tgt_id"]
        #         + dp["description"],
        #         "metadata": {
        #             "created_at": dp.get("metadata", {}).get("created_at", time.time())
        #         },
        #     }
        #     for dp in all_relationships_data
        # }
        print("Embedding Edges ...")
        print("Add des")
        data_for_vdb = {}
        for dp in all_relationships_data:
            list_des = list(dp["description"].split("<SEP>"))
            list_src = list(dp["source_id"].split("<SEP>"))
            for des_id in range(len(list_des)):
                des = list_des[des_id]
                data_for_vdb[compute_mdhash_id(dp["src_id"] + dp["tgt_id"] + des, prefix="rel-")] = {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "chunk_id" : list_src[des_id],
                "description" : des,
                "content":dp["src_id"] + " " + des + " " + dp["tgt_id"],
                "metadata": {
                    "created_at": dp.get("metadata", {}).get("created_at", time.time())
                },
            }
        # print("Embedding Edges ...")

        # data_for_vdb = {}
        # for dp in all_relationships_data:
        #     list_des = list(dp["description_list"].split("<SEP>"))
        #     for des_id in range(len(list_des)):
        #         des = list_des[des_id]
        #         data_for_vdb[compute_mdhash_id(dp["src_id"] + dp["tgt_id"] + des, prefix="rel-")] = {
        #         "src_id": dp["src_id"],
        #         "tgt_id": dp["tgt_id"],
        #         "content":dp["src_id"] + " " + des + " " + dp["tgt_id"],
        #         "metadata": {
        #             "created_at": dp.get("metadata", {}).get("created_at", time.time())
        #         },
        #     }
        await relationships_vdb.upsert(data_for_vdb)


        # get chunks data:
        chunks_data = self.text_chunks._data
        await chunks_vdb.upsert(chunks_data)


        # finalize the database

        tasks = [
            storage_inst.index_done_callback()
            for storage_inst in [  # type: ignore
                entities_vdb,
                relationships_vdb,
                chunks_vdb,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)


        print("Add new embedding sucessfully")

    def create_embedding_from_jsonl(self, embedding_func, embedding_name):
            loop = always_get_an_event_loop()
            loop.run_until_complete(
                self.a_create_embedding_from_jsonl(embedding_func, embedding_name)
            )

    async def a_create_embedding_from_jsonl(self,
                                     embedding_func,
                                     embedding_name):        
        # create db for entities, relationships and chunks:
        entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_ENTITIES + "_" + embedding_name
            ),
            embedding_func=embedding_func,
            meta_fields={"entity_name", "chunk_id", "description"},
        )

        relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_RELATIONSHIPS + "_" + embedding_name
            ),
            embedding_func=embedding_func,
            meta_fields={"src_id", "tgt_id", "chunk_id", "description"},
        )

        chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=make_namespace(
                self.namespace_prefix, NameSpace.VECTOR_STORE_CHUNKS + "_" + embedding_name
            ),
            embedding_func=embedding_func,
        )

        # Intialize the database:
        tasks = []

        for storage in (
                entities_vdb,
                relationships_vdb,
                chunks_vdb,
            ):
                if storage:
                    tasks.append(storage.initialize())

        await asyncio.gather(*tasks)
        import json
        # load entities information from entities extraction
        with open(os.path.join(self.working_dir, "vector_data/entities_extraction.jsonl"), "r") as f:
            entities_data_raw = [json.loads(line) for line in f]

        print("Preprocess Nodes from json...")
        # preprocess entities data
        entities_data_dict = {}
        for line in tqdm(entities_data_raw):
            entity_name = line["entity_name"]
            des = line["description"]
            chunk_id = line["chunk_id"]
            if entity_name in entities_data_dict:
               entities_data_dict[entity_name]["description"].append(des)
               entities_data_dict[entity_name]["chunk_ids"].append(chunk_id)
            else:
               entities_data_dict[entity_name] = {
                   "description" : [des],
                   "chunk_ids" : [chunk_id]
               }
            
        print("Embedding Nodes from json...")
        data_for_vdb = {}
        for entity_name, entity_data in tqdm(entities_data_dict.items()):
            list_des = entity_data["description"]
            list_src = entity_data["chunk_ids"]
            for des_id in range(len(list_des)):
                des = list_des[des_id]
                chunk_id = list_src[des_id]
                data_for_vdb[compute_mdhash_id(entity_name + des, prefix="ent-")] = {
                    "content": entity_name + " " + des,
                    "entity_name": entity_name,
                    "chunk_id" : chunk_id,
                    "description" : des
                }
        await entities_vdb.upsert(data_for_vdb)

        with open(os.path.join(self.working_dir, "vector_data/relations_extraction.jsonl"), "r") as f:
            relations_data_raw = [json.loads(line) for line in f]

        # preprocess entities data
        print("Preprocess Edges from json...")
        relations_data_dict = {}
        for line in tqdm(relations_data_raw):
            src_name = line["src_id"]
            tgt_name = line["tgt_id"]
            des = line["description"]
            chunk_id = line["chunk_id"]
            if (src_name, tgt_name) in relations_data_dict:
               entities_data_dict[(src_name, tgt_name)]["description"].append(des)
               entities_data_dict[(src_name, tgt_name)]["chunk_ids"].append(chunk_id)
            else:
               relations_data_dict[(src_name, tgt_name)] = {
                   "description" : [des],
                   "chunk_ids" : [chunk_id]
               }
        
        print("Embedding Edges from json...")
        data_for_vdb = {}
        for (src_id, tgt_id), relations_data in tqdm(relations_data_dict.items()):
            list_des = relations_data["description"]
            list_src = relations_data["source_id"]
            for des_id in range(len(list_des)):
                des = list_des[des_id]
                data_for_vdb[compute_mdhash_id(src_id + tgt_id + des, prefix="rel-")] = {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "chunk_id" : list_src[des_id],
                "content":src_id + " " + des + " " + tgt_id,
                "description" : des
            }

        await relationships_vdb.upsert(data_for_vdb)

        print("Embedding chunks...")
        # get chunks data:
        chunks_data = self.text_chunks._data
        await chunks_vdb.upsert(chunks_data)


        # finalize the database

        tasks = [
            storage_inst.index_done_callback()
            for storage_inst in [  # type: ignore
                entities_vdb,
                relationships_vdb,
                chunks_vdb,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)


        print("Add new embedding sucessfully")

    def new_retrieval(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        # system_prompt: str | None = None, # Not used directly by recall funcs
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Sync retrieval using direct recall functions, returning candidates and keywords.

        Args:
            query (str): The query string.
            param (QueryParam): Configuration parameters for retrieval.

        Returns:
            Tuple[List[Dict[str, Any]], List[str], List[str]]: A tuple containing:
                - A list of candidate node/edge dictionaries.
                - The extracted high-level keywords.
                - The extracted low-level keywords.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.anew_retrieval(query, param))

    async def anew_retrieval(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        # system_prompt: str | None = None, # Not used directly by recall funcs
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Async retrieval using direct recall functions, returning candidates and keywords.

        Args:
            query (str): The query string.
            param (QueryParam): Configuration parameters for retrieval.

        Returns:
            Tuple[List[Dict[str, Any]], List[str], List[str]]: A tuple containing:
                - A list of candidate node/edge dictionaries.
                - The extracted high-level keywords.
                - The extracted low-level keywords.
        """
        logger.info(f"Executing anew_retrieval with mode: {param.mode}")

        # Call the function from operate_old which now returns a tuple
        candidates, hl_keywords, ll_keywords = await kg_direct_recall(
            query=query,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entities_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            text_chunks_db=self.text_chunks,
            query_param=param,
            global_config=asdict(self),
            hashing_kv=self.llm_response_cache # Pass hashing_kv if needed
        )

        logger.info(f"anew_retrieval completed. Returning {len(candidates)} candidates, {len(hl_keywords)} HL keywords, {len(ll_keywords)} LL keywords.")

        # Return the tuple directly
        return candidates, hl_keywords, ll_keywords
    
    def _most_relevant_text_chunks_from_nodes(
        self,
        query: str,
        list_nodes: list[str],
        threshold : int
    ) -> dict[str, str]:

        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.a_most_relevant_text_chunks_from_nodes(query,list_nodes,threshold))

    async def a_most_relevant_text_chunks_from_nodes(
        self,
        query: str,
        list_nodes: list[str],
        threshold: int
    ) -> dict[str, str]:

        list_chunk_id  = []
        for node in list_nodes:
            list_chunk_id.append(await _most_relevant_text_chunks_from_nodes(query,
                                                    self.chunk_entity_relation_graph,
                                                    self.entities_vdb,
                                                    node,
                                                    threshold
                                                    ) 
            )


        results = {}
        for i in range(len(list_chunk_id)):
            results[list_nodes[i]] = list_chunk_id[i]

        return results


    def _most_relevant_text_chunks_from_edges(
        self,
        query: str,
        list_edges: list[(str, str)],
        threshold : int
    ) -> dict[str, str]:

        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.a_most_relevant_text_chunks_from_edges(query,list_edges,threshold))

    async def a_most_relevant_text_chunks_from_edges(
        self,
        query: str,
        list_edges: list[(str, str)],
        threshold: int
    ) -> dict[str, str]:

        list_chunk_id = [] 
        for head, tgt in list_edges:
            list_chunk_id.append(await _most_relevant_text_chunks_from_edges(query,
                                                        self.chunk_entity_relation_graph,
                                                        self.relationships_vdb,
                                                        head,
                                                        tgt,
                                                        threshold
                                                        )
            )

        results = {}
        for i in range(len(list_chunk_id)):
            results[(list_edges[i][0], list_edges[i][1])] = list_chunk_id[i]

        return results

    def multi_hop_retrieval(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        max_hops: int = 3,
        min_confidence_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.a_multi_hop_retrieval(query,param,max_hops,min_confidence_threshold))


    async def a_multi_hop_retrieval(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        max_hops: int = 3,
        min_confidence_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Multi-hop retrieval for complex queries by decomposing into sub-queries.
        
        This method follows this flow:
        1. Decompose complex query into initial sub-queries
        2. For each sub-query:
           - Retrieve relevant text
           - Use original query + past sub-queries with retrieved context to formulate next query
        3. Build comprehensive context from all retrieved information
        
        Args:
            query (str): The complex query string
            param (QueryParam): Configuration parameters for retrieval
            max_hops (int): Maximum number of retrieval hops allowed
            min_confidence_threshold (float): Threshold to determine if retrieval is sufficient
            
        Returns:
            List[Dict[str, Any]]: Final combined candidates after multi-hop retrieval
        """
        logger.info(f"Starting multi-hop retrieval for query: {query}")
        
        # Step 1: Decompose the complex query into initial sub-queries
        initial_sub_queries = await self.decompose_complex_query(query)
        logger.info(f"Initial query decomposition: {initial_sub_queries}")
        
        # Initial state
        all_candidates = []
        retrieval_history = []  # List of {question, context} dictionaries
        current_sub_query = initial_sub_queries[0] if initial_sub_queries else query
        if len(initial_sub_queries) == 1:
            top_k = 25
        else:
            top_k = 20        
        # Step 2: Process sub-queries sequentially, generating next query based on retrieved context
        hop_count = 0
        while hop_count < max_hops:
            hop_count += 1
            logger.info(f"Executing hop {hop_count} with sub-query: {current_sub_query}")
            
            # Retrieve information for current sub-query
            candidates, hl_keywords, ll_keywords = await self.anew_retrieval(
                query=current_sub_query,
                param=param
            )
            
            if not candidates:
                logger.info(f"No candidates found for sub-query in hop {hop_count}")
                
                # Try using a predefined sub-query if available
                if hop_count < len(initial_sub_queries):
                    current_sub_query = initial_sub_queries[hop_count]
                    logger.info(f"Using next predefined sub-query: {current_sub_query}")
                    continue
                else:
                    # No more predefined queries and no candidates found
                    logger.info("No more sub-queries available and no candidates found. Stopping.")
                    break
            
            # Select top candidates for this hop
            top_candidates = candidates[:top_k]
            all_candidates.extend(top_candidates)
            
            # Extract content from top candidates for context
            context_items = []
            for candidate in top_candidates:
                if candidate.get('retrieval_type') == 'node':
                    content = candidate.get('entity_name', '') + ": " + candidate.get('description', '')
                    # source_type = "Entity"
                    # source_id = candidate.get('id', '')
                else:  # edge
                    content = candidate.get('src_id', '') + " -> " + candidate.get('description', '') + " -> " + candidate.get('tgt_id', '') 
                    # source_type = "Relation"
                    # source_id = f"{candidate.get('src_id', '')}->{candidate.get('tgt_id', '')}"
                
                if content:
                    context_items.append(f" {content}")
            
            current_context = "\n".join(context_items)
            
            # Store this question with its retrieved context
            retrieval_history.append({
                "question": current_sub_query,
                "context": current_context
            })
            
            # If we've reached the last hop, stop
            if hop_count >= max_hops:
                logger.info(f"Reached maximum hop count ({max_hops}), stopping")
                break
            
            # Check if we have a predefined next query
            if hop_count < len(initial_sub_queries):
                # Use the next predefined sub-query, but enhance it with context from previous retrievals
                base_next_query = initial_sub_queries[hop_count]
                
                # Generate an enhanced version that incorporates previous context
                current_sub_query = await self._enhance_sub_query_with_context(
                    original_query=query,
                    base_sub_query=base_next_query, 
                    retrieval_history=retrieval_history
                )
                logger.info(f"Enhanced predefined sub-query: {current_sub_query}")
            else:
                # Generate the next sub-query based on original query and retrieval history
                current_sub_query = await self._generate_next_sub_query_from_context(
                    original_query=query,
                    retrieval_history=retrieval_history
                )
                
                # If no next query could be generated or we've answered the question fully
                if not current_sub_query:
                    logger.info("No further sub-queries needed, stopping")
                    break
                    
                logger.info(f"Generated next sub-query: {current_sub_query}")
        
        logger.info(f"Multi-hop retrieval completed after {hop_count} hops, found {len(all_candidates)} candidates")
        resorted_candidates = sorted(
            all_candidates, 
            key=lambda x: x.get("score", 0.0), 
            reverse=True
        )
        return resorted_candidates
    
    async def _enhance_sub_query_with_context(
        self,
        original_query: str,
        base_sub_query: str,
        retrieval_history: List[Dict[str, str]]
    ) -> str:
        """Enhance a predefined sub-query with context from previous retrievals.
        
        Args:
            original_query (str): The original complex query
            base_sub_query (str): The predefined next sub-query
            retrieval_history (List[Dict[str, str]]): History of previous retrievals
            
        Returns:
            str: Enhanced sub-query incorporating previous context
        """
        # Format the retrieval history
        history_text = ""
        for i, retrieval in enumerate(retrieval_history):
            history_text += f"Question {i+1}: {retrieval['question']}\n"
            history_text += f"Retrieved Context {i+1}:\n{retrieval['context']}\n\n"
        
        prompt = f"""
You are refining a sub-question based on previous retrievals.

Original Complex Question: {original_query}

Previous Retrievals:
{history_text}

Next Sub-Question (base version): {base_sub_query}

Your task: Refine the next sub-question to incorporate insights from the retrieved contexts.
Keep the same basic intent as the base version, but make it more specific and targeted
based on the information retrieved so far.

Respond with ONLY the refined sub-question, nothing else.
        """
        # print("Enhance prompt: ",prompt)
        # Generate a more stable and shorter cache key
        import hashlib
        history_hash = hashlib.md5(history_text.encode()).hexdigest()
        query_hash = hashlib.md5(base_sub_query.encode()).hexdigest()
        cache_key = f"enhance_{query_hash[:8]}_{history_hash[:8]}"
        
        # Try to get from cache first
        if self.llm_response_cache:
            cached_result = await self.llm_response_cache.get_by_id(cache_key)
            if cached_result and isinstance(cached_result, dict) and 'content' in cached_result:
                return cached_result['content']
        
        # Call LLM to enhance the query
        response = await self.llm_model_func(prompt)
        
        # Clean up the response
        enhanced_query = response.strip()
        # print("Enhanced query: ",enhanced_query)
        # Save to cache
        if self.llm_response_cache:
            await self.llm_response_cache.upsert({
                cache_key: {'content': enhanced_query}
            })
        
        return enhanced_query
    
    async def _generate_next_sub_query_from_context(
        self,
        original_query: str,
        retrieval_history: List[Dict[str, str]]
    ) -> str:
        """Generate the next sub-query based on the original query and retrieval history.
        
        Args:
            original_query (str): The original complex query
            retrieval_history (List[Dict[str, str]]): History of previous retrievals
            
        Returns:
            str: Next sub-query to ask, or empty string if no further queries needed
        """
        # Format the retrieval history
        history_text = ""
        for i, retrieval in enumerate(retrieval_history):
            history_text += f"Question {i+1}: {retrieval['question']}\n"
            history_text += f"Retrieved Context {i+1}:\n{retrieval['context']}\n\n"
        
        prompt = f"""
        You are generating the next sub-question in a multi-hop question answering process.
        
        Original Complex Question: {original_query}
        
        Previous Retrievals:
        {history_text}
        
        Your task: 
        1. Determine if we have enough information to answer the original complex question
        2. If we DON'T have enough information, generate a specific next sub-question
        3. If we DO have enough information to answer the original question, respond with "SUFFICIENT"
        4. For some question like where's A born or what's A occupation?. You must define the last question to answer the question what's job or what's place to explore about that place or that job

        The next sub-question should:
        - Address a specific aspect of the original question not covered by previous retrievals
        - Build logically on information from previous contexts
        - Be concise and focused on a single aspect
        
        Next Sub-Question (or SUFFICIENT if no more questions needed):
        """
        # print("Enhance prompt: ",prompt)
        # Generate a more stable and shorter cache key
        import hashlib
        query_hash = hashlib.md5(original_query.encode()).hexdigest()
        history_hash = hashlib.md5(history_text.encode()).hexdigest()
        cache_key = f"nextq_{query_hash[:8]}_{history_hash[:8]}"
        
        # Try to get from cache first
        if self.llm_response_cache:
            cached_result = await self.llm_response_cache.get_by_id(cache_key)
            if cached_result and isinstance(cached_result, dict) and 'content' in cached_result:
                cached_content = cached_result['content']
                return cached_content if cached_content != "SUFFICIENT" else ""
        
        # Call LLM to generate next query
        response = await self.llm_model_func(prompt)
        # print(response)
        # Clean up the response
        next_query = response.strip()
        
        # Check if we have sufficient information
        if next_query.upper() == "SUFFICIENT" or "SUFFICIENT" in next_query.upper():
            next_query = ""
            cache_content = "SUFFICIENT"
        else:
            cache_content = next_query
        
        # Save to cache
        if self.llm_response_cache:
            await self.llm_response_cache.upsert({
                cache_key: {'content': cache_content}
            })
        
        return next_query

    async def decompose_complex_query(
        self,
        query: str
    ) -> List[str]:
        """Decompose a complex query into simpler sub-queries.
        
        Args:
            query (str): The complex query to decompose
            
        Returns:
            List[str]: A list of sub-queries in logical order for multi-hop retrieval
        """
#         prompt = f"""
# You are tasked with breaking down complex questions into smaller, sequential sub-questions.

# Complex Question: {query}

# Decompose this into 2-4 simpler sub-questions that:
# - Are small and focused on specific aspects
# - For 
# - Must follow a strict sequential order, where each question builds directly on the previous one
# - Together form a step-by-step reasoning path to answer the original complex question
# - Create a logical progression where answering one sub-question provides necessary context for the next
# - For some question like where's A born or what's A occupation?. You must define the last question to answer the question what's job or what's place to explore about that place or that job
# - Don't need exactly 4 question        

# Format your response exactly like this:
# 1. [First basic sub-question addressing foundational aspects]
# 2. [Second sub-question that directly uses insights from the first question]
# 3. [Third sub-question that builds upon answers from questions 1 and 2]
# 4. [Final sub-question that integrates previous answers toward the complete solution]

# Remember: The sequence of questions must form a clear chain where each answer becomes input for the next question."""
        
#         prompt =f"""
# You are tasked with breaking down complex questions into smaller, sequential sub-questions.

# Complex Question: {query}

# Decompose this into 2-4 simpler sub-questions that:

# Are small and strictly focused on a single, specific aspect. Each sub-question must address only one distinct piece of information or task.
# Must follow a strict sequential order, where each question builds directly on the previous one.
# Together form a step-by-step reasoning path to answer the original complex question.
# Create a logical progression where answering one sub-question provides necessary context for the next.
# For some questions like "Where was A born?" or "What is A's occupation?", the final sub-question should explore further details about that place or job.
# Don't need exactly 4 questions.


# Format your response exactly like this:

# 1. [First basic sub-question addressing a foundational, single aspect]
# 2. [Second sub-question that directly uses insights from the first question, focusing on another single aspect]
# 3. [Third sub-question that builds upon answers from questions 1 and 2, focusing on a further single aspect]
# 4. [Final sub-question that integrates previous answers toward the complete solution, potentially exploring a final single aspect in detail]
# Remember: The sequence of questions must form a clear chain where each answer becomes input for the next question."""
#         # Use cache to avoid redundant LLM calls
        prompt = f"""
You are tasked with breaking down complex questions into smaller, sequential sub-questions.

Complex Question: {query}

Decompose this into 2-4 simpler sub-questions that:
- Are small and strictly focused on a single, specific aspect. Each sub-question must address only one distinct piece of information or task.
- Must primarily use the exact terminology, entities, and phrasing present in the original complex question. Avoid introducing new nouns, categorizations (like 'book', 'movie', 'person' etc. unless part of the original query), or assumptions not explicitly stated in the {query}. If an entity type is ambiguous (e.g., "Labyrinth"), the sub-questions should reflect this ambiguity rather than assigning a premature type.
- Must follow a strict sequential order, where each question builds directly on the previous one.
- Together form a step-by-step reasoning path to answer the original complex question.
- Create a logical progression where answering one sub-question provides necessary context for the next.
- For types of questions like "Where was A born?" or "What is A's occupation?", the final sub-question  the final sub-question should seek a definition, description, or further relevant details about that identified attribute (e.g., 'What does a [identified occupation] do?' or 'Tell me more about [identified place].').
- Don't need exactly 4 questions.

For example:
Query: Where is the man who won the Nobel Prize in Physics in 2023?
Decomposite query: ["Who is the man that won the Nobel Prize in Physics in 2023?", "Where is that man currently located or primarily based?"]

This is an example of a simple question but need more infomation like i have discussed above.
Query: Where was Ada Lovelace born?
Decomposite query: ["Where was Ada Lovelace born?", "Tell me more about that place (e.g., city or region).""]

The last example:
Query: what is the second largest country in asia
Decomposite query: ["What is the second largest country in Asia?", "Tell me more about that country (e.g., its geography, culture, or significance)."]

Format your response exactly like this:
1. [First basic sub-question addressing a foundational, single aspect, using terminology from the original query]
2. [Second sub-question that directly uses insights from the first question, focusing on another single aspect, using terminology from the original query]
3. [Third sub-question that builds upon answers from questions 1 and 2, focusing on a further single aspect, using terminology from the original query]
4. [Final sub-question that integrates previous answers toward the complete solution, potentially exploring a final single aspect in detail, using terminology from the original query]
Remember: The sequence of questions must form a clear chain where each answer becomes input for the next question, and the phrasing should stay as close as possible to the original query.        """
        cache_key = f"query_decomp_{hash(query)}_"
        
        # Try to get from cache first
        if self.llm_response_cache:
            cached_result = await self.llm_response_cache.get_by_id(cache_key)
            if cached_result and isinstance(cached_result, dict) and 'content' in cached_result:
                try:
                    cached_subqueries = json.loads(cached_result['content'])
                    if isinstance(cached_subqueries, list) and len(cached_subqueries) > 0:
                        return cached_subqueries
                except:
                    # If parsing fails, regenerate
                    pass
        
        # Call LLM to decompose the query
        response = await self.llm_model_func(prompt)
        
        try:
            # Parse the response using regex
            import re
            
            # Look for numbered items directly
            sub_queries = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', response + "\n1.", re.DOTALL)
            
            # Clean up the extracted sub-queries and remove brackets if present
            cleaned_queries = []
            for q in sub_queries:
                q = q.strip()
                if q:
                    # Remove brackets if the response follows the example format
                    bracket_match = re.match(r'\[(.*)\]', q)
                    if bracket_match:
                        q = bracket_match.group(1).strip()
                    cleaned_queries.append(q)
            
            sub_queries = cleaned_queries
            
            # If no sub-queries were found, fall back to using the original query
            if not sub_queries:
                sub_queries = [query]
            
            # Save to cache
            if self.llm_response_cache:
                await self.llm_response_cache.upsert({
                    cache_key: {'content': json.dumps(sub_queries)}
                })
            
            logger.info(f"Decomposed complex query into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error parsing query decomposition response: {e}")
            # Fall back to original query
            return [query]


# ######################### Page Rank #############################
    async def precompute_pagerank_data(self, force_use_existing_embeddings=False):
        """
        Precompute and cache necessary data for PageRank retrieval to avoid
        recalculating expensive operations on each query.
        
        Args:
            force_use_existing_embeddings: Nếu True, chỉ sử dụng các embeddings đã có trong vector database,
                                          không tính toán lại cho các entities/facts/chunks không tìm thấy.
        
        This should be called after data is loaded/indexed and before using PageRank retrieval.
        """
        logger.info("Precomputing data for PageRank retrieval...")
        
        # Store precomputed data in a dictionary
        self._pagerank_precomputed_data = {}
        
        # 1. Load the NetworkX graph from the graph storage
        G_nx = nx.Graph()
        
        # Get the underlying NetworkX graph from storage
        if hasattr(self.chunk_entity_relation_graph, "_graph"):
            G_nx = self.chunk_entity_relation_graph._graph
            self._pagerank_precomputed_data["graph"] = G_nx
        else:
            logger.warning("Graph storage does not expose '_graph' attribute. Creating new graph.")
            # Create graph from entity and relationships
            G_nx = await self._build_graph_from_entities_relationships()
            self._pagerank_precomputed_data["graph"] = G_nx
        
        # 2. Prepare all_chunk_data dictionary
        # all_chunk_data = {}
        # chunks =  self.get_all_chunks()
        
        # # Process all chunks and cache entity relationships
        # for chunk_id, chunk in chunks.items():
        #     # Get entities in this chunk from the graph
        #     entity_ids_in_chunk = await self.get_entities_in_chunk(chunk_id)
            
        #     all_chunk_data[chunk_id] = {
        #         "content": chunk.get("content", ""),
        #         "entity_ids_in_chunk": entity_ids_in_chunk
        #     }
            
        all_chunk_data = {}
        chunks = self.get_all_chunks()
        
        # Chuẩn bị danh sách các coroutines để xử lý song song
        tasks = []
        chunk_ids = []
        
        for chunk_id in chunks.keys():
            chunk_ids.append(chunk_id)
            tasks.append(self.get_entities_in_chunk(chunk_id))
        
        # Thực thi tất cả các truy vấn entities cùng lúc 
        results = await asyncio.gather(*tasks)
        
        # Xử lý kết quả
        for i, chunk_id in enumerate(chunk_ids):
            all_chunk_data[chunk_id] = {
                "content": chunks[chunk_id].get("content", ""),
                "entity_ids_in_chunk": results[i]
            }
        

        self._pagerank_precomputed_data["all_chunk_data"] = all_chunk_data
        
        # 3. Precompute edge/fact embeddings
        logger.info("Precomputing edge embeddings...")
        all_facts_with_embeddings = await self._compute_all_fact_node_embeddings(G_nx)
        
        # Nếu force_use_existing_embeddings = True, chỉ giữ lại các edges có embedding trong vector database
        if force_use_existing_embeddings:
            # Lấy danh sách các edges có embeddings
            edges_with_embeddings = set(all_facts_with_embeddings.keys())
            
            # Lọc đồ thị chỉ giữ các edges có embeddings
            edges_to_remove = []
            for u, v in G_nx.edges():
                edge_id = tuple(sorted((u, v)))
                if edge_id not in edges_with_embeddings:
                    edges_to_remove.append((u, v))
            
            if edges_to_remove:
                logger.info(f"Removing {len(edges_to_remove)} edges without embeddings from graph")
                G_nx.remove_edges_from(edges_to_remove)
        
        self._pagerank_precomputed_data["all_facts_with_embeddings"] = all_facts_with_embeddings
        
        # 4. Precompute passage/chunk embeddings
        logger.info("Precomputing chunk embeddings...")
        all_passage_ids = list(all_chunk_data.keys())
        all_passage_embeddings_map = await self._compute_all_passage_embeddings(all_chunk_data, all_passage_ids)
        
        # Nếu force_use_existing_embeddings = True, chỉ giữ lại các chunks có embedding trong vector database
        if force_use_existing_embeddings:
            # Lọc bỏ các chunks không có embeddings
            chunks_to_remove = []
            for chunk_id in all_chunk_data.keys():
                if chunk_id not in all_passage_embeddings_map:
                    chunks_to_remove.append(chunk_id)
            
            if chunks_to_remove:
                logger.info(f"Removing {len(chunks_to_remove)} chunks without embeddings from all_chunk_data")
                for chunk_id in chunks_to_remove:
                    del all_chunk_data[chunk_id]
        
        self._pagerank_precomputed_data["all_passage_embeddings_map"] = all_passage_embeddings_map
        
        # 5. Precompute synonyms if available
        synonym_data = {}
        # Implement based on how synonyms are stored in your system
        self._pagerank_precomputed_data["synonym_data"] = synonym_data
        
        logger.info("Precomputation for PageRank retrieval completed.")
        return self._pagerank_precomputed_data

    async def _build_graph_from_entities_relationships(self):
        """Build a NetworkX graph from entities and relationships in storage."""
        G_nx = nx.Graph()
        
        # Add nodes (entities)
        entities_data = await self.get_all_entities()
        for entity_id, entity_data in entities_data.items():
            G_nx.add_node(entity_id, **entity_data)
        
        # Add edges (relationships)
        relationships = await self.get_all_relationships()
        for rel_id, rel_data in relationships.items():
            if "src_id" in rel_data and "tgt_id" in rel_data:
                src_id = rel_data["src_id"]
                tgt_id = rel_data["tgt_id"]
                G_nx.add_edge(src_id, tgt_id, weight=1.0, **rel_data)
        
        return G_nx

    async def _compute_all_fact_embeddings(self, G_nx):
        """
        Lấy embeddings cho tất cả facts/edges trong đồ thị.
        Ưu tiên lấy từ vector database nếu có sẵn, chỉ tính toán khi cần thiết.
        """

        all_embeddings = {}
        edge_texts = []
        edge_ids = []
        edges_to_compute = []  # Danh sách các edges cần tính toán lại
        
        logger.info("Getting fact embeddings from relationships_vdb where available")
        
        # Kiểm tra xem relationships_vdb có dữ liệu không
        if hasattr(self.relationships_vdb, "client_storage") and self.relationships_vdb.client_storage:
            # Lấy ma trận embedding từ vector database
            storage_data = self.relationships_vdb.client_storage
            
            # Lấy tất cả relationship embeddings từ vector database
            relationship_embeddings = {}
            if "data" in storage_data and "matrix" in storage_data:
                try:
                    # Đọc matrix từ chuỗi base64 (nếu lưu dạng này)
                    if isinstance(storage_data["matrix"], str):
                        import base64
                        matrix_data = np.frombuffer(base64.b64decode(storage_data["matrix"]), dtype=np.float32)
                        embedding_dim = storage_data.get("embedding_dim", 1024)
                        matrix = matrix_data.reshape(-1, embedding_dim)
                    else:
                        matrix = storage_data["matrix"]
                    
                    # Map các embeddings vào relationship IDs
                    for i, rel_data in enumerate(storage_data["data"]):
                        if i < len(matrix) and "src_id" in rel_data and "tgt_id" in rel_data:
                            # Tạo edge_id tương tự như cách chúng ta tạo trong graph
                            src_id = rel_data["src_id"]
                            tgt_id = rel_data["tgt_id"]
                            edge_id = tuple(sorted((src_id, tgt_id)))  # Đảm bảo ID nhất quán cho đồ thị vô hướng
                            
                            # Lưu embedding cho edge này
                            relationship_embeddings[edge_id] = matrix[i]
                    
                    logger.info(f"Loaded {len(relationship_embeddings)} embeddings from relationships_vdb")
                except Exception as e:
                    logger.error(f"Error loading relationship embeddings from vector database: {e}")
        
        # Duyệt qua tất cả edges trong đồ thị
        for u, v in G_nx.edges():
            edge_id = tuple(sorted((u, v)))  # Đảm bảo ID nhất quán cho đồ thị vô hướng
            
            # Nếu đã có sẵn embedding cho edge này trong vector database
            if edge_id in relationship_embeddings:
                all_embeddings[edge_id] = relationship_embeddings[edge_id]
            else:
                # Nếu không có sẵn, chuẩn bị để tính toán
                edge_text = f"Relation between {u} and {v}"
                edge_texts.append(edge_text)
                edge_ids.append(edge_id)
                edges_to_compute.append((u, v))
        
        # Tính toán embeddings cho các edges còn lại nếu cần
        if edge_texts:
            logger.info(f"Computing embeddings for {len(edge_texts)} edges that are not in vector database")
            embeddings = await self.embedding_func(edge_texts)
            
            # Map edge IDs to embeddings
            for i, edge_id in enumerate(edge_ids):
                all_embeddings[edge_id] = embeddings[i]
        
        return all_embeddings

    async def _compute_all_fact_node_embeddings(self, G_nx):
        """
        Lấy embeddings cho tất cả facts/edges trong đồ thị.
        Ưu tiên lấy từ vector database nếu có sẵn, chỉ tính toán khi cần thiết.
        """

        all_embeddings = {}
        edge_texts = []
        edge_ids = []
        edges_to_compute = []  # Danh sách các edges cần tính toán lại
        
        logger.info("Getting fact embeddings from entities_vdb where available")
        
        # Kiểm tra xem relationships_vdb có dữ liệu không
        if hasattr(self.entities_vdb, "client_storage") and self.entities_vdb.client_storage:
            # Lấy ma trận embedding từ vector database
            storage_data = self.entities_vdb.client_storage
            
            # Lấy tất cả relationship embeddings từ vector database
            entities_embeddings = {}
            if "data" in storage_data and "matrix" in storage_data:
                try:
                    # Đọc matrix từ chuỗi base64 (nếu lưu dạng này)
                    if isinstance(storage_data["matrix"], str):
                        import base64
                        matrix_data = np.frombuffer(base64.b64decode(storage_data["matrix"]), dtype=np.float32)
                        embedding_dim = storage_data.get("embedding_dim", 1024)
                        matrix = matrix_data.reshape(-1, embedding_dim)
                    else:
                        matrix = storage_data["matrix"]
                    
                    # Map các embeddings vào relationship IDs
                    for i, rel_data in enumerate(storage_data["data"]):
                        if i < len(matrix):
                            # Tạo edge_id tương tự như cách chúng ta tạo trong graph
                            entity_name = rel_data["entity_name"]
                            des = rel_data["description"]
                            entity_name_des = (entity_name, des)  # Đảm bảo ID nhất quán cho đồ thị vô hướng
                            # Lưu embedding cho edge này
                            entities_embeddings[entity_name_des] = matrix[i]
                    
                    logger.info(f"Loaded {len(entities_embeddings)} embeddings from entities_vdb")
                except Exception as e:
                    logger.error(f"Error loading relationship embeddings from vector database: {e}")
        
        # # Duyệt qua tất cả edges trong đồ thị
        # for u, v in G_nx.edges():
        #     edge_id = tuple(sorted((u, v)))  # Đảm bảo ID nhất quán cho đồ thị vô hướng
            
        #     # Nếu đã có sẵn embedding cho edge này trong vector database
        #     if edge_id in relationship_embeddings:
        #         all_embeddings[edge_id] = relationship_embeddings[edge_id]
        #     else:
        #         # Nếu không có sẵn, chuẩn bị để tính toán
        #         edge_text = f"Relation between {u} and {v}"
        #         edge_texts.append(edge_text)
        #         edge_ids.append(edge_id)
        #         edges_to_compute.append((u, v))
        
        # # Tính toán embeddings cho các edges còn lại nếu cần
        # if edge_texts:
        #     logger.info(f"Computing embeddings for {len(edge_texts)} edges that are not in vector database")
        #     embeddings = await self.embedding_func(edge_texts)
            
        # Map edge IDs to embeddings
        # for i, edge_id in enumerate(edge_ids):
        #     all_embeddings[edge_id] = embeddings[i]

        all_embeddings = entities_embeddings
        
        return all_embeddings

    async def _compute_all_passage_embeddings(self, all_chunk_data, chunk_ids):
        """
        Lấy embeddings cho tất cả các passages/chunks.
        Ưu tiên lấy từ vector database nếu có sẵn, chỉ tính toán khi cần thiết.
        """
        result = {}
        chunks_to_compute = []
        passage_texts = []
        
        logger.info("Getting passage embeddings from chunks_vdb where available")
        
        # Kiểm tra xem chunks_vdb có dữ liệu không
        if hasattr(self.chunks_vdb, "client_storage") and self.chunks_vdb.client_storage:
            # Lấy ma trận embedding từ vector database
            storage_data = self.chunks_vdb.client_storage
            
            # Lấy tất cả chunk embeddings từ vector database
            chunk_embeddings = {}
            if "data" in storage_data and "matrix" in storage_data:
                try:
                    # Đọc matrix từ chuỗi base64 (nếu lưu dạng này)
                    if isinstance(storage_data["matrix"], str):
                        import base64
                        matrix_data = np.frombuffer(base64.b64decode(storage_data["matrix"]), dtype=np.float32)
                        embedding_dim = storage_data.get("embedding_dim", 1024)
                        matrix = matrix_data.reshape(-1, embedding_dim)
                    else:
                        matrix = storage_data["matrix"]
                    
                    # Map các embeddings vào chunk IDs
                    for i, chunk_data in enumerate(storage_data["data"]):
                        if i < len(matrix) and "__id__" in chunk_data:
                            chunk_id = chunk_data["__id__"]
                            # Lưu embedding cho chunk này
                            chunk_embeddings[chunk_id] = matrix[i]
                    
                    logger.info(f"Loaded {len(chunk_embeddings)} embeddings from chunks_vdb")
                except Exception as e:
                    logger.error(f"Error loading chunk embeddings from vector database: {e}")
        
        # Duyệt qua tất cả chunks cần embeddings
        for i, chunk_id in enumerate(chunk_ids):
            # Nếu đã có sẵn embedding cho chunk này trong vector database
            if chunk_id in chunk_embeddings:
                result[chunk_id] = chunk_embeddings[chunk_id]
            else:
                # Nếu không có sẵn, chuẩn bị để tính toán
                if chunk_id in all_chunk_data:
                    passage_texts.append(all_chunk_data[chunk_id].get("content", ""))
                else:
                    passage_texts.append("")  # Placeholder for missing chunks
                chunks_to_compute.append(chunk_id)
        
        # Tính toán embeddings cho các chunks còn lại nếu cần
        if chunks_to_compute:
            logger.info(f"Computing embeddings for {len(chunks_to_compute)} chunks that are not in vector database")
            embeddings = await self.embedding_func(passage_texts)
            
            # Map chunk IDs to embeddings
            for i, chunk_id in enumerate(chunks_to_compute):
                result[chunk_id] = embeddings[i]
        
        return result

    async def compute_query_embeddings(self, query, embedding_types=["passage", "fact"]):
        """
        Compute embeddings for query, optimized to support different embedding types
        while minimizing API calls.
        
        Args:
            query: The user's query string
            embedding_types: List of embedding types to compute (e.g., ["passage", "fact"])
        
        Returns:
            Dict mapping embedding types to their embeddings
        """
        embeddings = {}
        
        # Compute embeddings for all requested types
        query_embedding = await self.embedding_func([query])
        query_embedding = query_embedding[0]  # Get the first embedding
        
        # Use the same embedding for all types (if your model doesn't differentiate)
        # In a production system, you might have different embedding models for different types
        for embedding_type in embedding_types:
            embeddings[embedding_type] = query_embedding
        
        return embeddings

    async def retrieval_with_page_rank(
        self,
        query: str,
        config: dict = None,
        top_k: int = 5,
        use_precomputed_data: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Retrieves chunks using personalized PageRank for a given query.
        
        Args:
            query: The user's query string.
            config: Dictionary of hyperparameters for PageRank algorithm:
                - linking_top_k_facts: Number of top facts to consider.
                - passage_node_weight_factor: Factor to weigh DPR scores for personalization.
                - damping_factor: Damping factor for PageRank (e.g., 0.5).
                - num_retrieved_chunks: Number of chunks to return.
            top_k: Number of chunks to return.
            use_precomputed_data: Whether to use precomputed data (recommended for efficiency).
            
        Returns:
            List of tuples (chunk_id, score).
        """
        logger.info(f"Performing PageRank retrieval for query: '{query}'")
        
        if config is None:
            config = {
                "linking_top_k_facts": 10,
                "passage_node_weight_factor": 0.2,
                "damping_factor": 0.5,
                "num_retrieved_chunks": top_k,
                "use_synonyms": False,
                "synonym_weight_factor": 0.5,
                "direct_dpr_to_chunk_weight": 0.1,
                "average_ppr_for_chunk": True,
                "dpr_only_chunk_factor": 0.01,
            }
        
        # Check if we have precomputed data
        if use_precomputed_data and not hasattr(self, "_pagerank_precomputed_data"):
            logger.warning("Precomputed data not found. Running precomputation now.")
            await self.precompute_pagerank_data()
        
        # Get or compute required data
        if use_precomputed_data and hasattr(self, "_pagerank_precomputed_data"):
            # Use precomputed data
            G_nx = self._pagerank_precomputed_data["graph"]
            all_chunk_data = self._pagerank_precomputed_data["all_chunk_data"]
            all_facts_with_embeddings = self._pagerank_precomputed_data["all_facts_with_embeddings"]
            all_passage_embeddings_map = self._pagerank_precomputed_data["all_passage_embeddings_map"]
            synonym_data = self._pagerank_precomputed_data["synonym_data"]
        else:
            # Compute data on the fly (less efficient)
            G_nx = await self._build_graph_from_entities_relationships()
            
            all_chunk_data = {}
            chunks =  self.get_all_chunks()
            for chunk_id, chunk in chunks.items():
                entity_ids_in_chunk = await self.get_entities_in_chunk(chunk_id)
                all_chunk_data[chunk_id] = {
                    "content": chunk.get("content", ""),
                    "entity_ids_in_chunk": entity_ids_in_chunk
                }
            
            all_facts_with_embeddings = await self._compute_all_fact_node_embeddings(G_nx)
            all_passage_ids = list(all_chunk_data.keys())
            all_passage_embeddings_map = await self._compute_all_passage_embeddings(all_chunk_data, all_passage_ids)
            synonym_data = {}  # Placeholder
        
        # Get query embeddings
        query_embeddings = await self.compute_query_embeddings(query, ["fact", "passage"])
        query_embedding_for_fact = query_embeddings["fact"]
        query_embedding_for_passage = query_embeddings["passage"]
        logger.info("Query embeddings generated.")
        
        # Utility functions
        def calculate_similarity(vec1, vec2):
            """Calculate cosine similarity between two vectors."""
            vec1 = np.squeeze(vec1)
            vec2 = np.squeeze(vec2)
            if vec1.ndim == 0 or vec2.ndim == 0 or vec1.shape[0] != vec2.shape[0]:
                return 0.0
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            return dot_product / (norm_vec1 * norm_vec2)
            
        def min_max_normalize_dict_scores(scores_dict):
            """Normalize scores using min-max scaling."""
            if not scores_dict:
                return {}
            values = np.array(list(scores_dict.values()))
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            if range_val == 0:  # All values are equal
                return {k: 1.0 for k in scores_dict}
            return {k: (v - min_val) / range_val for k, v in scores_dict.items()}
        
        # 1. Fact Scoring and Simplified Reranking
        query_fact_scores = {}
        for fact_id, fact_emb in all_facts_with_embeddings.items():
            query_fact_scores[fact_id] = calculate_similarity(query_embedding_for_fact, fact_emb)
        
        # Sort facts and get top K
        sorted_facts = sorted(query_fact_scores.items(), key=lambda item: item[1], reverse=True)
        top_k_candidate_facts_with_scores = sorted_facts[:config.get('linking_top_k_facts', 10)]
        logger.info(f"Scored {len(all_facts_with_embeddings)} facts. Top {len(top_k_candidate_facts_with_scores)} facts selected.")
        
        # 2. Dense Passage Retrieval (DPR)
        dpr_scores = {}
        for passage_id, passage_emb in all_passage_embeddings_map.items():
            dpr_scores[passage_id] = calculate_similarity(query_embedding_for_passage, passage_emb)
        normalized_dpr_scores = min_max_normalize_dict_scores(dpr_scores)
        logger.info("Dense Passage Retrieval (DPR) scores calculated and normalized.")
        
        # 3. Prepare Personalization Vector for PageRank
        personalization_vector = {node: 0.0 for node in G_nx.nodes()}
        
        # 3a. Add weights from Facts/Edges
        if top_k_candidate_facts_with_scores:
            fact_contribution_scores = {fact_id: score for fact_id, score in top_k_candidate_facts_with_scores}
            normalized_fact_contribution_scores = min_max_normalize_dict_scores(fact_contribution_scores)
            
            count = 0
            for fact_id, normalized_score in normalized_fact_contribution_scores.items():
                u, v = fact_id
                if u in personalization_vector:
                    personalization_vector[u] += normalized_score
                # if v in personalization_vector:
                #     personalization_vector[v] += normalized_score
        logger.info(f"Number of nodes: {len([k for k, v in personalization_vector.items() if v > 0.0])}")
        for k, v in personalization_vector.items():
            if v > 0.0:
                logger.info(f"Node: {k} - {v}")
        logger.info("Weights from facts added to personalization vector.")
        
        # 3b. Add weights from Passage DPR scores
        passage_influence_factor = config.get('passage_node_weight_factor', 0.1)
        
        entity_to_dpr_passage_score_sum = defaultdict(float)
        entity_passage_counts = defaultdict(int)
        
        for chunk_id, chunk_detail in all_chunk_data.items():
            passage_dpr_score = normalized_dpr_scores.get(chunk_id, 0.0)
            for entity_id in chunk_detail.get('entity_ids_in_chunk', []):
                if entity_id in G_nx:
                    entity_to_dpr_passage_score_sum[entity_id] += passage_dpr_score
                    entity_passage_counts[entity_id] += 1
        
        for entity_id, total_score in entity_to_dpr_passage_score_sum.items():
            if entity_passage_counts[entity_id] > 0:
                avg_passage_score_for_entity = total_score / entity_passage_counts[entity_id]
                personalization_vector[entity_id] += avg_passage_score_for_entity * passage_influence_factor
        logger.info("Weights from DPR passage scores added to personalization vector.")
        
        # 3c. Synonym Expansion
        if config.get('use_synonyms', False) and synonym_data:
            nodes_with_initial_weights = [node for node, weight in personalization_vector.items() if weight > 1e-6]
            for node_id in nodes_with_initial_weights:
                original_weight = personalization_vector[node_id]
                for synonym_id in synonym_data.get(node_id, []):
                    if synonym_id in personalization_vector:
                        personalization_vector[synonym_id] += original_weight * config.get('synonym_weight_factor', 0.5)
            logger.info("Synonym expansion applied to personalization vector.")
        
        # Normalize personalization vector
        current_sum = sum(personalization_vector.values())
        print(current_sum)
        if current_sum > 1e-9:
            personalization_vector = {k: v / current_sum for k, v in personalization_vector.items()}
        else:
            logger.warning("Personalization vector is all zeros. Using uniform distribution.")
            if len(G_nx.nodes()) > 0:
                uniform_weight = 1.0 / len(G_nx.nodes())
                personalization_vector = {node: uniform_weight for node in G_nx.nodes()}
            else:
                logger.error("Graph has no nodes.")
                return []
        
        # 4. Run Personalized PageRank
        alpha = 1.0 - config.get('damping_factor', 0.5)
        logger.info(f"Running Personalized PageRank with alpha={alpha:.2f}...")
        

        sorted_personalization = sorted(list(personalization_vector.items()), key=lambda item: item[1], reverse=True)
        print(sorted_personalization[:10])
        if not G_nx.nodes():
            logger.warning("Graph is empty, cannot run PageRank.")
            return []
            
        try:
            ppr_scores = nx.pagerank(
                G_nx,
                alpha=alpha,
                personalization=personalization_vector,
                weight=None,
                tol=1.0e-6,
                max_iter=100
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge. Using results from last iteration.")
            ppr_scores = nx.pagerank(
                G_nx, 
                alpha=alpha, 
                personalization=personalization_vector, 
                weight=None, 
                tol=1.0e-4, 
                max_iter=500
            )
        
        logger.info("PageRank scores calculated for entity nodes.")
        
        sorted_ppr_scores = sorted(list(ppr_scores.items()), key=lambda item: item[1], reverse=True)
        print(sorted_ppr_scores[:10])
        # 5. Map PPR Entity Scores to Chunks and Rank Chunks
        chunk_final_scores = defaultdict(float)
        
        for chunk_id, chunk_detail in all_chunk_data.items():
            score_for_this_chunk = 0
            entities_in_this_chunk = chunk_detail.get('entity_ids_in_chunk', [])
            
            if not entities_in_this_chunk:
                # score_for_this_chunk = normalized_dpr_scores.get(chunk_id, 0.0) * config.get('dpr_only_chunk_factor', 0.1)
                score_for_this_chunk = 0

            else:
                num_entities_in_chunk_in_graph = 0
                for entity_id in entities_in_this_chunk:
                    if entity_id in ppr_scores:
                        score_for_this_chunk += ppr_scores.get(entity_id, 0.0)
                        num_entities_in_chunk_in_graph += 1
                
                if num_entities_in_chunk_in_graph > 0 and config.get('average_ppr_for_chunk', False):
                    score_for_this_chunk /= num_entities_in_chunk_in_graph
            
            # Add a portion of DPR score directly to chunk score
            # score_for_this_chunk += normalized_dpr_scores.get(chunk_id, 0.0) * config.get('direct_dpr_to_chunk_weight', 0.05)
            score_for_this_chunk += 0
            chunk_final_scores[chunk_id] = score_for_this_chunk
        
        sorted_chunks = sorted(chunk_final_scores.items(), key=lambda item: item[1], reverse=True)
        for chunk_id, score in sorted_chunks[:15]:
            print(normalized_dpr_scores.get(chunk_id, 0.0) * config.get('direct_dpr_to_chunk_weight', 0.05))
            print(f"Entities in the chunks{all_chunk_data[chunk_id].get('entity_ids_in_chunk', [])}")
        logger.info("PPR scores mapped to chunks and chunks ranked.")
        
        # 6. Return top N chunks
        retrieved_chunk_ids_with_scores = sorted_chunks[:config.get('num_retrieved_chunks', top_k)]
        logger.info(f"PageRank retrieval complete. Retrieved {len(retrieved_chunk_ids_with_scores)} chunks.")
        
        return retrieved_chunk_ids_with_scores

    async def get_all_entities(self):
        """Get all entities from the entity store."""
        try:
            # Try to get all entities from the VDB storage
            # Extract from NanoVectorDB's underlying storage
            if hasattr(self.entities_vdb, "client_storage") and self.entities_vdb.client_storage:
                entities = {}
                for entity_data in self.entities_vdb.client_storage.get("data", []):
                    entity_id = entity_data.get("__id__", "")
                    if entity_id:
                        # Convert data to dict format
                        entities[entity_id] = {
                            "entity_name": entity_data.get("entity_name", ""),
                            "entity_description": entity_data.get("description", ""),
                            "type": entity_data.get("type", ""),
                            # Add other relevant entity attributes
                        }
                return entities
        except Exception as e:
            logger.error(f"Error fetching entities: {e}")
            
        # Return empty dict if we can't fetch entities
        return {}

    async def get_all_relationships(self):
        """Get all relationships from the relationship store."""
        try:
            # Try to get all relationships from the VDB storage
            if hasattr(self.relationships_vdb, "client_storage") and self.relationships_vdb.client_storage:
                relationships = {}
                for rel_data in self.relationships_vdb.client_storage.get("data", []):
                    rel_id = rel_data.get("__id__", "")
                    if rel_id and "src_id" in rel_data and "tgt_id" in rel_data:
                        # Convert data to dict format with required src_id and tgt_id fields
                        try:
                            edge_data = await self.chunk_entity_relation_graph.get_edge_data(rel_data.get("src_id", ""), rel_data.get("tgt_id", ""))
                            weight = edge_data.get("weight", 1.0)
                        except Exception as e:
                            logger.error(f"Error fetching edge data: {e}")
                            weight = 1.0
                        relationships[rel_id] = {
                            "src_id": rel_data.get("src_id", ""),
                            "tgt_id": rel_data.get("tgt_id", ""),
                            "description": rel_data.get("description", ""),
                            "weight": weight,
                            # Add other relevant relationship attributes
                        }
                return relationships
        except Exception as e:
            logger.error(f"Error fetching relationships: {e}")
            
        # Return empty dict if we can't fetch relationships
        return {}

    def get_all_chunks(self):
        """Get all chunks from the chunk store."""
        try:
            # First try to get from the KV store
            chunks_data = self.text_chunks._data
            # print("getting chunks from KV store")
            if chunks_data:
                print("getting chunks from KV store")
                return chunks_data
                
            # If that fails, try from the vector database
            if hasattr(self.chunks_vdb, "client_storage") and self.chunks_vdb.client_storage:
                chunks = {}
                print("getting chunks from VDB")
                for chunk_data in self.chunks_vdb.client_storage.get("data", []):
                    chunk_id = chunk_data.get("__id__", "")
                    if chunk_id:
                        # Extract content and other relevant data
                        chunks[chunk_id] = {
                            "content": chunk_data.get("content", ""),
                            # Add other chunk attributes
                        }
                return chunks
        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
        
        # Return empty dict if we can't fetch chunks
        return {}

    async def get_entities_in_chunk(self, chunk_id):
        """Get all entities that appear in a specific chunk."""
        try:
            # Try to find connections in the graph storage
            entity_ids = []
            
            # 1. First check if we can get this information from the graph
            # If our graph has information about which entities are in which chunks
            if hasattr(self.chunk_entity_relation_graph, "_graph"):
                G_nx = self.chunk_entity_relation_graph._graph
                
                # Option 1: Check if any node has this chunk_id in its 'chunk_ids' attribute
                for node, node_data in G_nx.nodes(data=True):
                    if "source_id" in node_data and chunk_id in node_data.get("source_id", []):
                        entity_ids.append(node)
                        
                # Option 2: Look for special 'chunk' nodes or edges that connect chunks to entities
                # This depends on how your graph is structured - modify as needed
                # if not entity_ids:
                #     # If your graph has special structure where chunks are connected to entities
                #     # Example: Check if chunk_id is itself a node and get its neighbors
                #     if G_nx.has_node(chunk_id):
                #         entity_ids.extend(list(G_nx.neighbors(chunk_id)))
                
            
            # 2. If we couldn't find entities from the graph, try other storage methods
            # if not entity_ids:
            #     # Try to get this info from text_chunks metadata if available
            #     chunk_data = await self.text_chunks.get(chunk_id)
            #     if chunk_data and "entity_ids_in_chunk" in chunk_data:
            #         return chunk_data["entity_ids_in_chunk"]
                    
            return entity_ids
                
        except Exception as e:
            logger.error(f"Error fetching entities in chunk {chunk_id}: {e}")
            return []

    def retrieve_docs_with_pagerank(
        self,
        query: str,
        top_k: int = 5,
        pagerank_config: dict = None,
        return_docs: bool = True,
        use_precomputed_data: bool = True,
        force_use_existing_embeddings: bool = True,
    ) -> list[dict]:
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.a_retrieve_docs_with_pagerank(query,top_k,pagerank_config,return_docs,use_precomputed_data,force_use_existing_embeddings))

    async def a_retrieve_docs_with_pagerank(
        self,
        query: str,
        top_k: int = 5,
        pagerank_config: dict = None,
        return_docs: bool = True,
        use_precomputed_data: bool = True,
        force_use_existing_embeddings: bool = True,
    ) -> list[dict]:
        """
        High-level method to retrieve document chunks using PageRank algorithm.
        
        Args:
            query: User's text query
            top_k: Number of top document chunks to retrieve
            pagerank_config: Optional configuration parameters for PageRank algorithm
            return_docs: If True, return document contents; if False, just return IDs and scores
            use_precomputed_data: Whether to use precomputed data for faster retrieval
            force_use_existing_embeddings: If True, only use embeddings already in vector database
            
        Returns:
            List of document chunks with their content and relevance scores
        """
        # Check if we need to precompute data
        if use_precomputed_data and not hasattr(self, "_pagerank_precomputed_data"):
            logger.info("No precomputed data found. Running precomputation now...")
            await self.precompute_pagerank_data(force_use_existing_embeddings=force_use_existing_embeddings)
        
        # Set default PageRank configuration if not provided
        if pagerank_config is None:
            pagerank_config = {
                "linking_top_k_facts": 10,
                "passage_node_weight_factor": 0.2,
                "damping_factor": 0.5,
                "num_retrieved_chunks": top_k,
                "use_synonyms": False,
                "direct_dpr_to_chunk_weight": 0.1,
                "average_ppr_for_chunk": True,
                "dpr_only_chunk_factor": 0.01,
            }
        
        # Ensure num_retrieved_chunks matches top_k
        pagerank_config["num_retrieved_chunks"] = top_k
        
        # Call the PageRank retrieval method with option to use precomputed data
        chunk_ids_with_scores = await self.retrieval_with_page_rank(
            query=query,
            config=pagerank_config,
            top_k=top_k,
            use_precomputed_data=use_precomputed_data,
        )
        
        # If we just want IDs and scores, convert to dict format and return
        if not return_docs:
            return [{"id": chunk_id, "score": score} for chunk_id, score in chunk_ids_with_scores]
        
        # Otherwise, fetch the full document contents
        results = []
        
        # Use precomputed chunk data if available
        all_chunk_data = {}
        if use_precomputed_data and hasattr(self, "_pagerank_precomputed_data"):
            all_chunk_data = self._pagerank_precomputed_data.get("all_chunk_data", {})
        
        for chunk_id, score in chunk_ids_with_scores:
            try:
                # First check if we have this chunk in precomputed data
                if chunk_id in all_chunk_data and use_precomputed_data:
                    chunk_content = all_chunk_data[chunk_id].get("content", "")
                    result = {
                        "id": chunk_id,
                        "score": score,
                        "content": chunk_content,
                        "metadata": {
                            "entity_ids_in_chunk": all_chunk_data[chunk_id].get("entity_ids_in_chunk", [])
                        }
                    }
                    results.append(result)
                    continue
                    
                # If not in precomputed data, retrieve from storage
                chunk_data = await self.text_chunks._data.get(chunk_id)
                
                if chunk_data:
                    # Add the chunk data and score to results
                    result = {
                        "id": chunk_id,
                        "score": score,
                        "content": chunk_data.get("content", ""),
                        "metadata": {k: v for k, v in chunk_data.items() if k != "content"},
                    }
                    results.append(result)
                else:
                    # If chunk not found in text_chunks, try to find it from chunks_vdb
                    # This is a fallback in case the chunk is only in the vector DB
                    if hasattr(self.chunks_vdb, "client_storage") and self.chunks_vdb.client_storage:
                        for chunk_data in self.chunks_vdb.client_storage.get("data", []):
                            if chunk_data.get("__id__") == chunk_id:
                                result = {
                                    "id": chunk_id,
                                    "score": score,
                                    "content": chunk_data.get("content", ""),
                                    "metadata": {k: v for k, v in chunk_data.items() 
                                               if k not in ["content", "__id__", "__vector__", "__metrics__"]},
                                }
                                results.append(result)
                                break
                    else:
                        # If we can't find the chunk content anywhere, just include the ID and score
                        logger.warning(f"Could not find content for chunk ID: {chunk_id}")
                        results.append({"id": chunk_id, "score": score, "content": "", "metadata": {}})
                        
            except Exception as e:
                logger.error(f"Error retrieving content for chunk {chunk_id}: {e}")
                results.append({"id": chunk_id, "score": score, "content": "", "metadata": {}})
                
        return results

########################
    # def retrieve_docs_with_enhanced_method(
    #     self,
    #     query: str,
    #     top_k: int = 50,
    #     a: float = 0.7,
    #     b: float = 0.3,
    #     query_param: QueryParam = None,
    #     use_precomputed_data: bool = True,
    #     force_use_existing_embeddings: bool = True,
    # ) -> list[dict]:
    #     """
    #     High-level method to retrieve document chunks using the enhanced retrieval method.
        
    #     Args:
    #         query: User's text query
    #         top_k: Number of top document chunks to retrieve
    #         a: Weight for embedding similarity score
    #         b: Weight for average node score 
    #         query_param: Parameters controlling the retrieval process
    #         use_precomputed_data: Whether to use precomputed data for faster retrieval
    #         force_use_existing_embeddings: If True, only use embeddings already in vector database
            
    #     Returns:
    #         List of document chunks with their content and relevance scores
    #     """
    #     loop = always_get_an_event_loop()
        
    #     if query_param is None:
    #         query_param = QueryParam(
    #             top_k=top_k,
    #             mode="local",  # Focus on nodes
    #             unique_entity_edge=False
    #         )
        
    #     return loop.run_until_complete(
    #         self.a_retrieve_docs_with_enhanced_method(
    #             query=query,
    #             top_k=top_k,
    #             a=a,
    #             b=b,
    #             query_param=query_param,
    #             use_precomputed_data=use_precomputed_data,
    #             force_use_existing_embeddings=force_use_existing_embeddings
    #         )
    #     )

    # async def a_retrieve_docs_with_enhanced_method(
    #     self,
    #     query: str,
    #     top_k: int = 5,
    #     a: float = 0.7,
    #     b: float = 0.3,
    #     query_param: QueryParam = None,
    #     use_precomputed_data: bool = True,
    #     force_use_existing_embeddings: bool = True,
    # ) -> list[dict]:
    #     """Async version of retrieve_docs_with_enhanced_method"""
        
    #     # Check for precomputed data
    #     if use_precomputed_data and not hasattr(self, "_pagerank_precomputed_data"):
    #         logger.info("No precomputed data found. Running precomputation now...")
    #         await self.precompute_pagerank_data(force_use_existing_embeddings=force_use_existing_embeddings)
        
    #     if query_param is None:
    #         query_param = QueryParam(
    #             top_k=top_k,
    #             mode="local",  # Focus on nodes
    #             unique_entity_edge=True
    #         )
        
    #     # Set default PageRank configuration if needed
    #     global_config = {}  # Add your global config here
        
    #     # Call the enhanced retrieval method
    #     result_chunks = await enhanced_chunk_retrieval(
    #         query=query,
    #         knowledge_graph_inst=self.chunk_entity_relation_graph,
    #         entities_vdb=self.entities_vdb,
    #         relationships_vdb=self.relationships_vdb,
    #         text_chunks_db=self.text_chunks,
    #         chunks_vdb=self.chunks_vdb,
    #         embedding_func=self.embedding_func,
    #         query_param=query_param,
    #         global_config=global_config,
    #         a=a,
    #         b=b,
    #         top_k=top_k,
    #         use_precomputed_data=use_precomputed_data
    #     )
        
    #     return result_chunks
    

    def retrieve_docs_with_enhanced_method(
        self,
        query: str,
        top_k: int = 5,
        weight_chunk: float = 1,
        weight_entity: float = 0.3,
        weight_edge: float = 0.3,
        query_param: QueryParam = None,
        use_precomputed_data: bool = True,
        force_use_existing_embeddings: bool = True,
        threshold: float = 0.7,
        method = None
    ) -> list[dict]:
        """
        High-level method to retrieve document chunks using the enhanced retrieval method.
        
        Args:
            query: User's text query
            top_k: Number of top document chunks to retrieve
            a: Weight for embedding similarity score
            b: Weight for average node score 
            query_param: Parameters controlling the retrieval process
            use_precomputed_data: Whether to use precomputed data for faster retrieval
            force_use_existing_embeddings: If True, only use embeddings already in vector database
            
        Returns:
            List of document chunks with their content and relevance scores
        """
        loop = always_get_an_event_loop()
        
        if query_param is None:
            query_param = QueryParam(
                top_k=top_k,
                mode="local",  # Focus on nodes
                unique_entity_edge=True
            )
        
        return loop.run_until_complete(
            self.a_retrieve_docs_with_enhanced_method(
                query=query,
                top_k=top_k,
                a=weight_chunk,
                b=weight_entity,
                c=weight_edge,
                # query_param=query_param,
                use_precomputed_data=use_precomputed_data,
                force_use_existing_embeddings=force_use_existing_embeddings,
                method = method,
                threshold=threshold
            )
        )

    async def a_retrieve_docs_with_enhanced_method(
        self,
        query: str,
        top_k: int = 5,
        a: float = 0.7,
        b: float = 0.3,
        c: float = 0.3,
        query_param: QueryParam = None,
        use_precomputed_data: bool = True,
        force_use_existing_embeddings: bool = True,
        method=None,
        threshold: float = 0.7
    ) -> list[dict]:
        """Async version of retrieve_docs_with_enhanced_method"""
        
        # Check for precomputed data
        # if use_precomputed_data and not hasattr(self, "_pagerank_precomputed_data"):
        #     logger.info("No precomputed data found. Running precomputation now...")
        #     await self.precompute_pagerank_data(force_use_existing_embeddings=force_use_existing_embeddings)
        
        if query_param is None:
            query_param = QueryParam(
                top_k=top_k,
                mode="local",  # Focus on nodes
                unique_entity_edge=True
            )
        
        # Set default PageRank configuration if needed
        global_config = {}  # Add your global config here
        
        # Call the enhanced retrieval method
        result_chunks = await enhanced_chunk_retrieval_direct(
            query=query,
            knowledge_graph_inst=self.chunk_entity_relation_graph,
            entities_vdb=self.entities_vdb,
            relationships_vdb=self.relationships_vdb,
            text_chunks_db=self.text_chunks,
            chunks_vdb=self.chunks_vdb,
            embedding_func=self.embedding_func,
            chunk_dict = self.text_chunks._data,
            doc_dict= self.full_docs._data,
            # query_param=query_param,
            # global_config=global_config,
            a=a,
            b=b,
            c=c,
            top_k_entities=top_k,
            top_k_edges=top_k,
            top_k_chunks=top_k,
            use_precomputed_data=use_precomputed_data,
            method=method,
            relevance_threshold=threshold
        )
        
        return result_chunks