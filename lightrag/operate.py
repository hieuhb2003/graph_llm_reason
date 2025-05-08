from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator
from collections import Counter, defaultdict
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    statistic_data,
    get_conversation_turns,
    csv_string_to_list,
    detect_language,
    load_json,
    write_json
)
import numpy as np
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS, get_prompt
import time
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import os
import glob


def chunking_by_token_size(
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = decode_tokens_by_tiktoken(
                            _tokens[start : start + max_token_size],
                            model_name=tiktoken_model,
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = decode_tokens_by_tiktoken(
                tokens[start : start + max_token_size], model_name=tiktoken_model
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = get_prompt("summarize_entity_descriptions", language)
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    current_language: str = "Vietnamese"
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
        language=current_language
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    # description = GRAPH_FIELD_SEP.join(
    #     sorted(set([dp["description"] for dp in nodes_data] + already_description))
    # )
    # source_id = GRAPH_FIELD_SEP.join(
    #     set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    # )

    description = GRAPH_FIELD_SEP.join(
        [dp["description"] for dp in nodes_data] + already_description
    )
    
    source_id = GRAPH_FIELD_SEP.join(
        [dp["source_id"] for dp in nodes_data] + already_source_ids
    )
    # description = await _handle_entity_relation_summary(
    #     entity_name, description, global_config
    # )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        language=detect_language(description),
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_language = None

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 0.0 if missing
            already_weights.append(already_edge.get("weight", 0.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    split_string_by_multi_markers(
                        already_edge["source_id"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.append(already_edge["description"])

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )
                
            # Get language if it already exists
            if already_edge.get("language") is not None:
                already_language = already_edge["language"]

    # Process edges_data with None checks
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    # description = GRAPH_FIELD_SEP.join(
    #     sorted(
    #         set(
    #             [dp["description"] for dp in edges_data if dp.get("description")]
    #             + already_description
    #         )
    #     )
    # )

    description = GRAPH_FIELD_SEP.join(
                [dp["description"] for dp in edges_data if dp.get("description")]
                + already_description
    )
    keywords = GRAPH_FIELD_SEP.join(
                [dp["keywords"] for dp in edges_data if dp.get("keywords")]
                + already_keywords
    )
    # keywords = GRAPH_FIELD_SEP.join(
    #     sorted(
    #         set(
    #             [dp["keywords"] for dp in edges_data if dp.get("keywords")]
    #             + already_keywords
    #         )
    #     )
    # )
    # source_id = GRAPH_FIELD_SEP.join(
    #     set(
    #         [dp["source_id"] for dp in edges_data if dp.get("source_id")]
    #         + already_source_ids
    #     )
    # )
    source_id = GRAPH_FIELD_SEP.join(
            [dp["source_id"] for dp in edges_data if dp.get("source_id")]
            + already_source_ids
    )

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                    "language": detect_language(description),
                },
            )
    # description = await _handle_entity_relation_summary(
    #     f"({src_id}, {tgt_id})", description, global_config
    # )
    
    # Detect language from description if not already set
    if not already_language:
        language = detect_language(description)
    else:
        language = already_language
    
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            language=language,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        language=language,
    )

    return edge_data


async def save_data_to_json_files(
    entities_data: dict[str, dict[str, Any]] = None,
    relationships_data: dict[str, dict[str, Any]] = None,
    chunks_data: dict[str, dict[str, Any]] = None,
    working_dir: str = None,
    namespace: str = None,
) -> dict[str, str]:
    """Save entity, relationship, and chunk data to JSON files for later vector DB insertion
    
    Args:
        entities_data: Dictionary mapping entity IDs to entity data
        relationships_data: Dictionary mapping relationship IDs to relationship data
        chunks_data: Dictionary mapping chunk IDs to chunk data
        working_dir: Directory to save JSON files
        namespace: Namespace to use in filenames
        
    Returns:
        dict[str, str]: Paths to the saved JSON files
    """
    if working_dir is None:
        working_dir = os.getcwd()
    
    if namespace is None:
        namespace = "default"
        
    timestamp = int(time.time())
    result_paths = {}
    
    # Create a directory for the JSON files if it doesn't exist
    json_dir = os.path.join(working_dir, "vector_data")
    os.makedirs(json_dir, exist_ok=True)
    
    # Save entities data
    if entities_data:
        entities_file = os.path.join(json_dir, f"entities.json")
        write_json(entities_data, entities_file)
        result_paths["entities"] = entities_file
        logger.info(f"Saved {len(entities_data)} entities to {entities_file}")
    
    # Save relationships data
    if relationships_data:
        relationships_file = os.path.join(json_dir, f"relationships.json")
        write_json(relationships_data, relationships_file)
        result_paths["relationships"] = relationships_file
        logger.info(f"Saved {len(relationships_data)} relationships to {relationships_file}")
    
    # Save chunks data
    if chunks_data:
        chunks_file = os.path.join(json_dir, f"chunks.json")
        write_json(chunks_data, chunks_file)
        result_paths["chunks"] = chunks_file
        logger.info(f"Saved {len(chunks_data)} chunks to {chunks_file}")
    
    # Save a manifest file that lists all the files saved in this batch
    manifest = {
        "timestamp": timestamp,
        "namespace": namespace,
        "files": result_paths,
        "counts": {
            "entities": len(entities_data) if entities_data else 0,
            "relationships": len(relationships_data) if relationships_data else 0,
            "chunks": len(chunks_data) if chunks_data else 0,
        }
    }
    
    manifest_file = os.path.join(json_dir, f"manifest.json")
    write_json(manifest, manifest_file)
    result_paths["manifest"] = manifest_file
    
    return result_paths

async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    llm_response_cache: BaseKVStorage | None = None,
) -> BaseGraphStorage | None:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    enable_llm_cache_for_entity_extract: bool = global_config[
        "enable_llm_cache_for_entity_extract"
    ]
    # Check if we're delaying vector DB updates
    delay_vector_db_update = global_config.get("delay_vector_db_update", False)

    # Set up directories for JSONL files if delay_vector_db_update is True
    if delay_vector_db_update:
        working_dir = global_config.get("working_dir", os.getcwd())
        vector_data_dir = os.path.join(working_dir, "vector_data")
        os.makedirs(vector_data_dir, exist_ok=True)
        
        # Define entity and relation JSONL files
        entity_jsonl_file = os.path.join(vector_data_dir, "entities_extraction.jsonl")
        relation_jsonl_file = os.path.join(vector_data_dir, "relations_extraction.jsonl")

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    _type = get_prompt("DEFAULT_ENTITY_TYPES", language)
    entity_types = global_config["addon_params"].get(
        "entity_types", _type
    )
    example_number = global_config["addon_params"].get("example_number", None)
    _examples = get_prompt("entity_extraction_examples", language)
    if example_number and example_number < len(_examples):
        examples = "\n".join(
            _examples[: int(example_number)]
        )
    else:
        examples = "\n".join(_examples)

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"], 
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = get_prompt("entity_extraction", language)
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    continue_prompt = get_prompt("entiti_continue_extraction", language)
    if_loop_prompt = get_prompt("entity_extraction", language)

    already_processed = 0
    already_entities = 0
    already_relations = 0
    
    # Function to save entity to JSONL
    async def save_entity_to_jsonl(entity_data):
        if not delay_vector_db_update:
            return
        
        # print("Entity data: ", entity_data)
        
        # Create entity record with entity name, description and chunk_id
        entity_record = {
            "entity_name": entity_data["entity_name"],
            "description": entity_data["description"],
            "chunk_id": entity_data["source_id"],
            "timestamp": time.time()
        }
        
        # Append to JSONL file
        with open(entity_jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entity_record, ensure_ascii=False) + '\n')
    
    # Function to save relation to JSONL
    async def save_relation_to_jsonl(relation_data):
        if not delay_vector_db_update:
            return
        
        # print("Relation data: ", relation_data)
        # Create relation record with src, tgt, description and chunk_id
        relation_record = {
            "src_id": relation_data["src_id"],
            "tgt_id": relation_data["tgt_id"],
            "description": relation_data["description"],
            "chunk_id": relation_data["source_id"],
            "timestamp": time.time()
        }
        
        # Append to JSONL file
        with open(relation_jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(relation_record, ensure_ascii=False) + '\n')

    async def _user_llm_func_with_cache(
        input_text: str, history_messages: list[dict[str, str]] = None
    ) -> str:
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            if history_messages:
                history = json.dumps(history_messages, ensure_ascii=False)
                _prompt = history + "\n" + input_text
            else:
                _prompt = input_text

            arg_hash = compute_args_hash(_prompt)
            cached_return, _1, _2, _3 = await handle_cache(
                llm_response_cache,
                arg_hash,
                _prompt,
                "default",
                cache_type="extract",
                force_llm_cache=True,
            )
            if cached_return:
                logger.debug(f"Found cache for {arg_hash}")
                statistic_data["llm_cache"] += 1
                return cached_return
            statistic_data["llm_call"] += 1
            if history_messages:
                res: str = await use_llm_func(
                    input_text, history_messages=history_messages
                )
            else:
                res: str = await use_llm_func(input_text)
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type="extract",
                ),
            )
            return res

        if history_messages:
            return await use_llm_func(input_text, history_messages=history_messages)
        else:
            return await use_llm_func(input_text)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """ "Prpocess a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunck-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        """
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        current_language = global_config.get("addon_params", {}).get("current_language", "Vietnamese")
    
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await _user_llm_func_with_cache(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _user_llm_func_with_cache(
                continue_prompt, history_messages=history
            )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await _user_llm_func_with_cache(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, current_language
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                # Save entity to JSONL immediately if delay_vector_db_update is True
                if delay_vector_db_update:
                    await save_entity_to_jsonl(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                if_relation["language"] = current_language
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
                # Save relation to JSONL immediately if delay_vector_db_update is True
                if delay_vector_db_update:
                    await save_relation_to_jsonl(if_relation)
                    
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        logger.debug(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
        )
        return dict(maybe_nodes), dict(maybe_edges)

    tasks = [_process_single_content(c) for c in ordered_chunks]
    results = await asyncio.gather(*tasks)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )

    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )

    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    # Prepare data for vector databases or JSON files
    entities_for_vdb = {
        compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
            "content": dp["entity_name"] + dp["description"],
            "entity_name": dp["entity_name"],
        }
        for dp in all_entities_data
    }
    
    relationships_for_vdb = {
        compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
            "src_id": dp["src_id"],
            "tgt_id": dp["tgt_id"],
            "content": dp["keywords"]
            + dp["src_id"]
            + dp["tgt_id"]
            + dp["description"],
            "metadata": {
                "created_at": dp.get("metadata", {}).get("created_at", time.time())
            },
        }
        for dp in all_relationships_data
    }

    # Update vector databases if not delaying updates
    if not delay_vector_db_update and entity_vdb is not None and relationships_vdb is not None:
        await entity_vdb.upsert(entities_for_vdb)
        await relationships_vdb.upsert(relationships_for_vdb)

    # Log message about saved JSONL files if delay_vector_db_update is True
    if delay_vector_db_update:
        logger.info(f"Entity extraction data saved to {entity_jsonl_file}")
        logger.info(f"Relation extraction data saved to {relation_jsonl_file}")

    return knowledge_graph_inst

async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return get_prompt("fail_response", language)
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    print(ll_keywords_str)
    print(hl_keywords_str)
    # Build context
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return get_prompt("fail_response", language)

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = system_prompt if system_prompt else get_prompt("rag_response", language) 
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response

async def kg_retrieval(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    docs_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return []
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build context
    chunk_list = await _build_retrieval_context(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        chunks_vdb,
        text_chunks_db,
        docs_db,
        query_param,
    )

    return chunk_list

async def naive_retrieval(query: str,
                          query_param: QueryParam,
                          global_config: dict[str, str],
                          chunks_vdb: BaseVectorStorage,
                          text_chunks_db: BaseKVStorage):

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return get_prompt("fail_response", language)

    chunks_ids = [r["id"] for r in results]
    chunks_distance = [float(r["distance"]) for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)


    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    valid_chunks = [chunk["content"] for chunk in valid_chunks]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return get_prompt("fail_response", language)

    chunk_scores_dict = {}
    for i in range(len(valid_chunks)):
        chunk_scores_dict[valid_chunks[i]] = chunks_distance[i]

    return chunk_scores_dict


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """

    # 1. Handle cache if needed - add cache type for keywords
    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    _examples = get_prompt("keywords_extraction_examples", language)
    if example_number and example_number < len(_examples):
        examples = "\n".join(
            _examples[: int(example_number)]
        )
    else:
        examples = "\n".join(_examples)


    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = get_prompt("keywords_extraction", language).format(
        query=text, examples=examples, language=language, history=history_context
    )

    len_of_prompts = len(encode_string_by_tiktoken(kw_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    # 5. Call the LLM for keyword extraction
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 7. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )
    return hl_keywords, ll_keywords


async def mix_kg_vector_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    """
    Hybrid retrieval implementation combining knowledge graph and vector search.

    This function performs a hybrid search by:
    1. Extracting semantic information from knowledge graph
    2. Retrieving relevant text chunks through vector similarity
    3. Combining both results for comprehensive answer generation
    """
    # 1. Cache handling
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash("mix", query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "mix", cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    # 2. Execute knowledge graph and vector searches in parallel
    async def get_kg_context():
        try:
            # Extract keywords using extract_keywords_only function which already supports conversation history
            hl_keywords, ll_keywords = await extract_keywords_only(
                query, query_param, global_config, hashing_kv
            )

            if not hl_keywords and not ll_keywords:
                logger.warning("Both high-level and low-level keywords are empty")
                return None

            # Convert keyword lists to strings
            ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
            hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

            # Set query mode based on available keywords
            if not ll_keywords_str and not hl_keywords_str:
                return None
            elif not ll_keywords_str:
                query_param.mode = "global"
            elif not hl_keywords_str:
                query_param.mode = "local"
            else:
                query_param.mode = "hybrid"

            # Build knowledge graph context
            context = await _build_query_context(
                ll_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )

            return context

        except Exception as e:
            logger.error(f"Error in get_kg_context: {str(e)}")
            return None

    async def get_vector_context():
        # Consider conversation history in vector search
        augmented_query = query
        if history_context:
            augmented_query = f"{history_context}\n{query}"

        try:
            # Reduce top_k for vector search in hybrid mode since we have structured information from KG
            mix_topk = min(10, query_param.top_k)
            results = await chunks_vdb.query(augmented_query, top_k=mix_topk)
            if not results:
                return None

            chunks_ids = [r["id"] for r in results]
            chunks = await text_chunks_db.get_by_ids(chunks_ids)

            valid_chunks = []
            for chunk, result in zip(chunks, results):
                if chunk is not None and "content" in chunk:
                    # Merge chunk content and time metadata
                    chunk_with_time = {
                        "content": chunk["content"],
                        "created_at": result.get("created_at", None),
                    }
                    valid_chunks.append(chunk_with_time)

            if not valid_chunks:
                return None

            maybe_trun_chunks = truncate_list_by_token_size(
                valid_chunks,
                key=lambda x: x["content"],
                max_token_size=query_param.max_token_for_text_unit,
            )

            if not maybe_trun_chunks:
                return None

            # Include time information in content
            formatted_chunks = []
            for c in maybe_trun_chunks:
                chunk_text = c["content"]
                if c["created_at"]:
                    chunk_text = f"[Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['created_at']))}]\n{chunk_text}"
                formatted_chunks.append(chunk_text)

            logger.debug(
                f"Truncate chunks from {len(chunks)} to {len(formatted_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
            )
            return "\n--New Chunk--\n".join(formatted_chunks)
        except Exception as e:
            logger.error(f"Error in get_vector_context: {e}")
            return None

    # 3. Execute both retrievals in parallel
    kg_context, vector_context = await asyncio.gather(
        get_kg_context(), get_vector_context()
    )

    # 4. Merge contexts
    if kg_context is None and vector_context is None:
        return get_prompt("fail_response", language)

    if query_param.only_need_context:
        return {"kg_context": kg_context, "vector_context": vector_context}

    # 5. Construct hybrid prompt
    sys_prompt = (
        system_prompt
        if system_prompt
        else get_prompt("mix_rag_response", language).format(
            kg_context=kg_context
            if kg_context
            else "No relevant knowledge graph information found",
            vector_context=vector_context
            if vector_context
            else "No relevant text information found",
            response_type=query_param.response_type,
            history=history_context,
        )
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[mix_kg_vector_query]Prompt Tokens: {len_of_prompts}")

    # 6. Generate response
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # 清理响应内容
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        # 7. Save cache - 只有在收集完整响应后才缓存
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode="mix",
                cache_type="query",
            ),
        )

    return response


async def _build_query_context(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    if query_param.mode == "local":
        entities_context, relations_context, text_units_context, entity_chunks_mapping = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context, relation_chunks_mapping = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    else:  # hybrid mode
        ll_data, hl_data = await asyncio.gather(
            _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            ),
            _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            ),
        )

        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
            ll_entity_chunks_mapping,
        ) = ll_data

        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
            hl_relation_chunks_mapping,
        ) = hl_data

        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    # not necessary to use LLM to generate a response
    if not entities_context.strip() and not relations_context.strip():
        return None
    
    #---------------------------------------------------
    
    
    # entities_list = csv_string_to_list(entities_context)
    # relations_list = csv_string_to_list(relations_context)
    # text_units_list = csv_string_to_list(text_units_context)
    
    # # Extract raw text chunks from the CSV
    # raw_text_chunks = []
    # if text_units_list and len(text_units_list) > 1:  # Skip header row
    #     for row in text_units_list[1:]:
    #         if len(row) > 1:  # Ensure row has content column
    #             raw_text_chunks.append(row[1]) 
    #             print(row[1])
    #             print("---------------------------------------------------")

    #---------------------------------------------------
    
    result = f"""
    -----Entities-----
    ```csv
    {entities_context}
    ```
    -----Relationships-----
    ```csv
    {relations_context}
    ```
    -----Sources-----
    ```csv
    {text_units_context}
    ```
    """.strip()
    return result

async def _build_retrieval_context(
    raw_query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    docs_db: BaseKVStorage,
    query_param: QueryParam,
):
    if query_param.mode == "local":
        entities_context, relations_context, text_units_context, entity_chunks_mapping = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context, relation_chunks_mapping = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    else:  # hybrid mode
        ll_data, hl_data = await asyncio.gather(
            _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            ),
            _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            ),
        )

        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
            entity_chunks_mapping,
        ) = ll_data

        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
            relation_chunks_mapping,
        ) = hl_data

    # async def calculate_scores(query, keyword, chunks_mapping, vdb, chunk_db, docs_db, text_chunks_db, is_entity=True):
    #     # Get query embedding once
    #     query_embedding = await vdb.embedding_func([query])
    #     query_embedding = query_embedding[0].reshape(1, -1)  # Reshape for sklearn

    #     key_embedding = await vdb.embedding_func([keyword])
    #     key_embedding = key_embedding[0].reshape(1, -1)
        
    #     # Flatten data structure for processing all chunks together
    #     all_keys = []
    #     all_chunks = []
    #     all_chunk_ids = []
    #     chunk_to_key_map = {}  # Map each chunk to its key
    #     chunk_index_map = {}   # Track original index of each chunk
        
    #     # Collect all data in flat structures
    #     for key, chunks in chunks_mapping.items():
    #         for chunk in chunks:
    #             all_keys.append(key)
    #             all_chunks.append(chunk)
    #             chunk_id = compute_mdhash_id(chunk, prefix="chunk-")
    #             all_chunk_ids.append(chunk_id)
    #             chunk_to_key_map[chunk] = key
                
    #     # Batch load all chunk metadata in one call
    #     all_chunk_metadata = await asyncio.gather(*[text_chunks_db.get_by_id(chunk_id) for chunk_id in all_chunk_ids])
        
    #     # Extract all doc_ids
    #     all_doc_ids = []
    #     for metadata in all_chunk_metadata:
    #         doc_id = metadata.get("full_doc_id") if metadata else None
    #         all_doc_ids.append(doc_id)
            
    #     # Batch load all document content
    #     all_doc_contents = await asyncio.gather(*[
    #         docs_db.get_by_id(doc_id) if doc_id else None 
    #         for doc_id in all_doc_ids
    #     ])
        
    #     # Build list of valid chunks and their contents
    #     valid_chunks = []
    #     valid_keys = []
    #     valid_contents = []
        
    #     for i, chunk in enumerate(all_chunks):
    #         doc_content = None
    #         if all_doc_contents[i] and "content" in all_doc_contents[i]:
    #             doc_content = all_doc_contents[i]["content"]
    #         else:
    #             doc_content = chunk
                
    #         if not doc_content:
    #             continue
                
    #         valid_chunks.append(chunk)
    #         valid_keys.append(all_keys[i])
    #         valid_contents.append(doc_content)
            
    #     if not valid_chunks:
    #         return []
            
    #     # Batch load or calculate all embeddings at once
        
    #     # 1. First, get unique keys and their embeddings
    #     unique_keys = list(set(valid_keys))
    #     key_embeddings = {}
        
    #     for key in unique_keys:
    #         if is_entity:
    #             embedding = await vdb.get_entity_embedding(key)
    #             if embedding is None:
    #                 print(f"Entity embedding for {key} not found, computing...")
    #                 embedding = (await vdb.embedding_func([key]))[0]
    #         else:
    #             if "----" in key:
    #                 src_id, tgt_id = key.split("----", 1)
    #                 embedding = await vdb.get_relation_embedding(src_id, tgt_id)
    #             else:
    #                 embedding = None
                    
    #             if embedding is None:
    #                 print(f"Relation embedding for {key} not found, computing...")
    #                 embedding = (await vdb.embedding_func([key]))[0]
            
    #         key_embeddings[key] = embedding
            
    #     # 2. Get or compute chunk embeddings
    #     chunk_embeddings = await asyncio.gather(*[chunk_db.get_chunk_embedding(compute_mdhash_id(chunk, prefix="chunk-")) for chunk in valid_chunks])
        
    #     # 3. Compute any missing chunk embeddings
    #     chunks_to_compute = []
    #     chunks_to_compute_indices = []
        
    #     for i, embedding in enumerate(chunk_embeddings):
    #         if embedding is None:
    #             chunks_to_compute.append(valid_chunks[i])
    #             chunks_to_compute_indices.append(i)
                
    #     if chunks_to_compute:
    #         computed_embeddings = await chunk_db.embedding_func(chunks_to_compute)
    #         for idx, embedding in zip(chunks_to_compute_indices, computed_embeddings):
    #             chunk_embeddings[idx] = embedding
                
    #     # 4. Build combined matrices for matrix operations
    #     # Create matrix structure: each row has [key_embedding, chunk_embedding]
    #     matrix_rows = []
    #     for i, chunk in enumerate(valid_chunks):
    #         key = valid_keys[i]
    #         key_emb = key_embeddings[key]
    #         chunk_emb = chunk_embeddings[i]
    #         matrix_rows.append((valid_contents[i], key_emb, chunk_emb))
            
    #     # 5. Perform matrix operations
    #     if matrix_rows:
    #         # Extract content and embeddings
    #         contents = [row[0] for row in matrix_rows]
    #         key_embs = np.array([row[1] for row in matrix_rows])
    #         chunk_embs = np.array([row[2] for row in matrix_rows])
            
    #         # Calculate similarities in one batch operation
    #         key_similarities = sklearn_cosine_similarity(key_embedding, key_embs)[0]
    #         content_similarities = sklearn_cosine_similarity(query_embedding, chunk_embs)[0]
            
    #         # Calculate final scores
    #         total_scores = 0.4 * key_similarities + 0.6 * content_similarities
            
    #         # Create result tuples and convert scores to native Python float for JSON compatibility
    #         scored_docs = [(content, float(score)) for content, score in zip(contents, total_scores)]
            
    #         # Sort by scores
    #         scored_docs.sort(key=lambda x: x[1], reverse=True)
    #         return scored_docs
            
    #     return []
    
    
    
    async def calculate_scores(query, keyword, chunks_mapping, vdb, chunk_db, docs_db, text_chunks_db, is_entity=True):
        import time
        import logging
        
        # Thiết lập logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("performance_log")
        
        # Hàm tiện ích để ghi nhận thời gian
        def log_time(start_time, step_name):
            elapsed = time.time() - start_time
            logger.info(f"THỜI GIAN - {step_name}: {elapsed:.4f} giây")
            return time.time()
        
        total_start = time.time()
        
        # 1. Thực hiện embedding query và keyword cùng lúc
        step_start = time.time()
        query_embedding = await vdb.embedding_func([query])
        key_embedding = await vdb.embedding_func([keyword])
        # query_embedding, key_embedding = await asyncio.gather(
        #     vdb.embedding_func([query]), 
        #     vdb.embedding_func([keyword])
        # )
        query_embedding = query_embedding[0].reshape(1, -1)
        key_embedding = key_embedding[0].reshape(1, -1)
        step_start = log_time(step_start, "Embedding query và keyword")
        
        # 2. Chuẩn bị dữ liệu
        all_keys = []
        all_chunks = []
        all_chunk_ids = []
        
        # Tạo danh sách các chunk_ids trước để batch processing
        for key, chunks in chunks_mapping.items():
            for chunk in chunks:
                all_keys.append(key)
                all_chunks.append(chunk)
                chunk_id = compute_mdhash_id(chunk, prefix="chunk-")
                all_chunk_ids.append(chunk_id)
        
        step_start = log_time(step_start, "Chuẩn bị dữ liệu chunks")
        logger.info(f"Số lượng chunks: {len(all_chunks)}")
        
        # 3. Tối ưu hóa bằng cách gom nhóm các truy vấn database
        # Lấy metadata chunks và embeddings cùng lúc
        step_start = time.time()
        chunk_metadata_task = asyncio.gather(*[text_chunks_db.get_by_id(chunk_id) for chunk_id in all_chunk_ids])
        chunk_embeddings_task = asyncio.gather(*[chunk_db.get_chunk_embedding(chunk_id) for chunk_id in all_chunk_ids])
        
        all_chunk_metadata, initial_chunk_embeddings = await asyncio.gather(chunk_metadata_task, chunk_embeddings_task)
        step_start = log_time(step_start, "Lấy metadata và embeddings của chunks")
        
        # 4. Trích xuất doc_ids và batch load tất cả document contents
        step_start = time.time()
        all_doc_ids = [metadata.get("full_doc_id") if metadata else None for metadata in all_chunk_metadata]
        all_doc_contents = await asyncio.gather(*[
            docs_db.get_by_id(doc_id) if doc_id else None 
            for doc_id in all_doc_ids
        ])
        step_start = log_time(step_start, "Lấy nội dung documents")
        
        # 5. Chuẩn bị dữ liệu hợp lệ và xác định chunk embeddings nào cần tính toán
        step_start = time.time()
        valid_chunks = []
        valid_keys = []
        valid_contents = []
        valid_chunk_embeddings = []
        chunks_to_compute = []
        chunks_to_compute_indices = []
        
        for i, chunk in enumerate(all_chunks):
            # Lấy nội dung
            doc_content = all_doc_contents[i].get("content", chunk) if all_doc_contents[i] else chunk
            if not doc_content:
                continue
                
            valid_chunks.append(chunk)
            valid_keys.append(all_keys[i])
            valid_contents.append(doc_content)
            
            # Kiểm tra embedding
            if initial_chunk_embeddings[i] is None:
                chunks_to_compute.append(chunk)
                chunks_to_compute_indices.append(len(valid_chunk_embeddings))
            
            valid_chunk_embeddings.append(initial_chunk_embeddings[i])
        
        if not valid_chunks:
            logger.info(f"Tổng thời gian xử lý: {time.time() - total_start:.4f} giây - Không có chunk hợp lệ")
            return []
        
        logger.info(f"Số lượng chunks hợp lệ: {len(valid_chunks)}")
        logger.info(f"Số lượng chunks cần tính embedding: {len(chunks_to_compute)}")
        step_start = log_time(step_start, "Chuẩn bị dữ liệu hợp lệ")
        
        # 6. Batch load unique key embeddings
        step_start = time.time()
        unique_keys = list(set(valid_keys))
        key_embedding_tasks = []
        
        for key in unique_keys:
            if is_entity:
                task = vdb.get_entity_embedding(key)
            else:
                if "----" in key:
                    src_id, tgt_id = key.split("----", 1)
                    task = vdb.get_relation_embedding(src_id, tgt_id)
                else:
                    task = None
            key_embedding_tasks.append(task)
        
        key_embedding_results = await asyncio.gather(*key_embedding_tasks)
        step_start = log_time(step_start, "Lấy embeddings cho keys")
        
        # 7. Tính toán các embeddings còn thiếu
        step_start = time.time()
        missing_key_indices = [i for i, emb in enumerate(key_embedding_results) if emb is None]
        keys_to_compute = [unique_keys[i] for i in missing_key_indices]
        
        if keys_to_compute:
            logger.info(f"Số lượng keys cần tính embedding: {len(keys_to_compute)}")
            computed_key_embeddings = await vdb.embedding_func(keys_to_compute)
            for i, embedding in zip(missing_key_indices, computed_key_embeddings):
                key_embedding_results[i] = embedding
        
        # Lưu kết quả vào dictionary để truy cập nhanh
        key_embeddings = {unique_keys[i]: emb for i, emb in enumerate(key_embedding_results)}
        step_start = log_time(step_start, "Tính toán embeddings thiếu cho keys")
        
        # 8. Tính toán chunk embeddings còn thiếu
        step_start = time.time()
        if chunks_to_compute:
            computed_embeddings = await chunk_db.embedding_func(chunks_to_compute)
            for idx, embedding in zip(chunks_to_compute_indices, computed_embeddings):
                valid_chunk_embeddings[idx] = embedding
        step_start = log_time(step_start, "Tính toán embeddings thiếu cho chunks")
        
        # 9. Chuẩn bị ma trận cho vector operations
        step_start = time.time() 
        if valid_chunks:
            # Chuyển đổi dữ liệu thành numpy arrays
            key_embs = np.array([key_embeddings[key] for key in valid_keys])
            chunk_embs = np.array(valid_chunk_embeddings)
            
            # Tính toán similarities cùng lúc
            key_similarities = sklearn_cosine_similarity(key_embedding, key_embs)[0]
            content_similarities = sklearn_cosine_similarity(query_embedding, chunk_embs)[0]
            
            # Sử dụng numpy để tính toán tổng điểm
            total_scores = 0.4 * key_similarities + 0.6 * content_similarities
            
            # Tạo kết quả và sắp xếp
            scored_docs = [(content, float(score)) for content, score in zip(valid_contents, total_scores)]
            result = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
            step_start = log_time(step_start, "Tính toán similarity và sắp xếp kết quả")
            log_time(total_start, "TỔNG THỜI GIAN XỬ LÝ")
            
            return result
        
        log_time(total_start, "TỔNG THỜI GIAN XỬ LÝ")
        return []
    
    ll_scored_chunks, hl_scored_chunks = [], []
    
    if query_param.mode == "hybrid":
        # Chạy cả hai tác vụ song song trong chế độ hybrid
        ll_task, hl_task = await asyncio.gather(
            calculate_scores(
                query=raw_query,
                keyword=ll_keywords,
                chunks_mapping=entity_chunks_mapping, 
                vdb=entities_vdb, 
                chunk_db=chunks_vdb,
                docs_db=docs_db,
                text_chunks_db=text_chunks_db,
                is_entity=True
            ),
            calculate_scores(
                query=raw_query,
                keyword=hl_keywords,
                chunks_mapping=relation_chunks_mapping, 
                vdb=relationships_vdb, 
                chunk_db=chunks_vdb,
                docs_db=docs_db,
                text_chunks_db=text_chunks_db,
                is_entity=False
            )
        )
        ll_scored_chunks, hl_scored_chunks = ll_task, hl_task
    elif query_param.mode == "local":
        ll_scored_chunks = await calculate_scores(
            query=raw_query,
            keyword=ll_keywords,
            chunks_mapping=entity_chunks_mapping, 
            vdb=entities_vdb, 
            chunk_db=chunks_vdb,
            docs_db=docs_db,
            text_chunks_db=text_chunks_db,
            is_entity=True
        )
    elif query_param.mode == "global":
        hl_scored_chunks = await calculate_scores(
            query=raw_query,
            keyword=hl_keywords,
            chunks_mapping=relation_chunks_mapping, 
            vdb=relationships_vdb, 
            chunk_db=chunks_vdb,
            docs_db=docs_db,
            text_chunks_db=text_chunks_db,
            is_entity=False
        )
    
    return (ll_scored_chunks, hl_scored_chunks)
    



async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query}, top_k: {query_param.top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )
    results = await entities_vdb.query(query, top_k=query_param.top_k)


    if not len(results):
        return "", "", ""
    # get entity information
    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        ),
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk
    use_text_units, use_relations = await asyncio.gather(
        _find_most_related_text_unit_from_entities(
            node_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
        _find_most_related_edges_from_entities(
            node_datas, query_param, knowledge_graph_inst
        ),
    )

    entity_chunks_mapping = {}
    for node in node_datas:
        entity_name = node["entity_name"]
        entity_source_ids = split_string_by_multi_markers(node.get("source_id", ""), [GRAPH_FIELD_SEP])
        
        # Lấy các chunks liên quan đến entity này
        entity_chunks = []
        for text_unit in use_text_units:
            # text_unit là dictionary chứa trực tiếp content
            entity_chunks.append(text_unit["content"])
        
        # Thêm vào mapping nếu có chunks
        if entity_chunks:
            entity_chunks_mapping[entity_name] = entity_chunks[:10]
            
    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} chunks"
    )

    # build prompt
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(use_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context, entity_chunks_mapping


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    tasks = []
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                tasks.append((c_id, index, this_edges))

    results = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id, _, _ in tasks]
    )

    for (c_id, index, this_edges), data in zip(tasks, results):
        all_text_units_lookup[c_id] = {
            "data": data,
            "order": index,
            "relation_counts": 0,
        }

        if this_edges:
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.debug(
        f"Truncate chunks from {len(all_text_units_lookup)} to {len(all_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack, all_edges_degree = await asyncio.gather(
        asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]),
        asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
        ),
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    logger.debug(
        f"Truncate relations from {len(all_edges)} to {len(all_edges_data)} (max tokens:{query_param.max_token_for_global_context})"
    )

    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas, edge_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                for r in results
            ]
        ),
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "rank": d,
            "created_at": k.get("__created_at__", None),  # 从 KV 存储中获取时间元数据
            **v,
        }
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    # edge_datas = sorted(
    #     edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    # )
    print("no_sort")
    len_edge_datas = len(edge_datas)
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    logger.debug(
        f"Truncate relations from {len_edge_datas} to {len(edge_datas)} (max tokens:{query_param.max_token_for_global_context})"
    )

    use_entities, use_text_units = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
    )


    relation_chunks_mapping = {}
    for edge in edge_datas:
        # Sử dụng description của relation làm key
        # relation_key = edge.get("description", "").strip()
        # if not relation_key:  # Nếu không có description, tạo key từ src_id và tgt_id
        relation_key = f"{edge['src_id']}----{edge['tgt_id']}"
            
        # Lấy các chunks liên quan đến relation này
        relation_chunks = []
        for text_unit in use_text_units:
            # text_unit là dictionary chứa trực tiếp content
            relation_chunks.append(text_unit["content"])
        
        # Thêm vào mapping nếu có chunks
        if relation_chunks:
            relation_chunks_mapping[relation_key] = relation_chunks[:10]

    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "Unknown")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context, relation_chunks_mapping


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[
                knowledge_graph_inst.get_node(entity_name)
                for entity_name in entity_names
            ]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.node_degree(entity_name)
                for entity_name in entity_names
            ]
        ),
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    logger.debug(
        f"Truncate chunks from {len(valid_text_units)} to {len(truncated_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources


async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return get_prompt("fail_response", language)

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return get_prompt("fail_response", language)

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return get_prompt("fail_response", language)

    logger.debug(
        f"Truncate chunks from {len(chunks)} to {len(maybe_trun_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return section

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = system_prompt if system_prompt else get_prompt("rag_response", language)
    sys_prompt = sys_prompt_temp.format(
        content_data=section,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[naive_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )

    return response


async def kg_query_with_keywords(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Refactored kg_query that does NOT extract keywords by itself.
    It expects hl_keywords and ll_keywords to be set in query_param, or defaults to empty.
    Then it uses those to build context and produce a final LLM response.
    """

    # ---------------------------
    # 1) Handle potential cache for query results
    # ---------------------------

    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # ---------------------------
    # 2) RETRIEVE KEYWORDS FROM query_param
    # ---------------------------

    # If these fields don't exist, default to empty lists/strings.
    hl_keywords = getattr(query_param, "hl_keywords", []) or []
    ll_keywords = getattr(query_param, "ll_keywords", []) or []

    # If neither has any keywords, you could handle that logic here.
    if not hl_keywords and not ll_keywords:
        logger.warning(
            "No keywords found in query_param. Could default to global mode or fail."
        )
        return get_prompt("fail_response", language)
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty, switching to global mode.")
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty, switching to local mode.")
        query_param.mode = "local"

    # Flatten low-level and high-level keywords if needed
    ll_keywords_flat = (
        [item for sublist in ll_keywords for item in sublist]
        if any(isinstance(i, list) for i in ll_keywords)
        else ll_keywords
    )
    hl_keywords_flat = (
        [item for sublist in hl_keywords for item in sublist]
        if any(isinstance(i, list) for i in hl_keywords)
        else hl_keywords
    )

    # Join the flattened lists
    ll_keywords_str = ", ".join(ll_keywords_flat) if ll_keywords_flat else ""
    hl_keywords_str = ", ".join(hl_keywords_flat) if hl_keywords_flat else ""

    # ---------------------------
    # 3) BUILD CONTEXT
    # ---------------------------
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )
    if not context:
        return get_prompt("fail_response", language)

    # If only context is needed, return it
    if query_param.only_need_context:
        return context

    # ---------------------------
    # 4) BUILD THE SYSTEM PROMPT + CALL LLM
    # ---------------------------

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = get_prompt("rag_response", language)
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query_with_keywords]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response

async def load_json_files_to_vector_db(
    vector_data_dir: str,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    manifest_pattern: str = None,
    namespace: str = None,
) -> dict[str, int]:
    """Load JSON files saved from prior extraction and create vector databases
    
    Args:
        vector_data_dir: Directory containing the JSON files
        entities_vdb: Vector database for entities
        relationships_vdb: Vector database for relationships  
        chunks_vdb: Vector database for chunks
        manifest_pattern: Pattern to match manifest files (default: all manifest files)
        namespace: Namespace to filter manifests by
        
    Returns:
        dict[str, int]: Counts of loaded entities, relationships, and chunks
    """
    if not os.path.exists(vector_data_dir):
        logger.error(f"Vector data directory '{vector_data_dir}' does not exist")
        return {"entities": 0, "relationships": 0, "chunks": 0}
    
    # Find all manifest files
    manifest_files = []
    if manifest_pattern:
        import glob
        manifest_files = glob.glob(os.path.join(vector_data_dir, manifest_pattern))
    else:
        manifest_files = [
            os.path.join(vector_data_dir, f) 
            for f in os.listdir(vector_data_dir)
            if f.startswith("manifest_") and f.endswith(".json")
        ]
    
    if namespace:
        manifest_files = [f for f in manifest_files if f"_{namespace}_" in f]
    
    if not manifest_files:
        logger.warning(f"No manifest files found in '{vector_data_dir}'")
        return {"entities": 0, "relationships": 0, "chunks": 0}
    
    logger.info(f"Found {len(manifest_files)} manifest files to process")
    
    # Process each manifest
    total_entities = 0
    total_relationships = 0
    total_chunks = 0
    
    for manifest_file in sorted(manifest_files):
        manifest = load_json(manifest_file)
        if not manifest:
            logger.warning(f"Empty or invalid manifest file: {manifest_file}")
            continue
        
        # Load entity data
        if "entities" in manifest["files"]:
            entity_file = manifest["files"]["entities"]
            if os.path.exists(entity_file):
                entity_data = load_json(entity_file)
                if entity_data:
                    total_entities += len(entity_data)
                    logger.info(f"Loading {len(entity_data)} entities from {entity_file}")
                    await entities_vdb.upsert(entity_data)
            else:
                logger.warning(f"Entity file {entity_file} not found")
        
        # Load relationship data
        if "relationships" in manifest["files"]:
            relationship_file = manifest["files"]["relationships"]
            if os.path.exists(relationship_file):
                relationship_data = load_json(relationship_file)
                if relationship_data:
                    total_relationships += len(relationship_data)
                    logger.info(f"Loading {len(relationship_data)} relationships from {relationship_file}")
                    await relationships_vdb.upsert(relationship_data)
            else:
                logger.warning(f"Relationship file {relationship_file} not found")
        
        # Load chunk data
        if "chunks" in manifest["files"]:
            chunk_file = manifest["files"]["chunks"]
            if os.path.exists(chunk_file):
                chunk_data = load_json(chunk_file)
                if chunk_data:
                    total_chunks += len(chunk_data)
                    logger.info(f"Loading {len(chunk_data)} chunks from {chunk_file}")
                    await chunks_vdb.upsert(chunk_data)
            else:
                logger.warning(f"Chunk file {chunk_file} not found")
    
    # Final counts
    result = {
        "entities": total_entities,
        "relationships": total_relationships,
        "chunks": total_chunks,
    }
    
    logger.info(f"Total loaded: {total_entities} entities, {total_relationships} relationships, {total_chunks} chunks")
    return result