from __future__ import annotations
import numpy as np
import asyncio
import json
import re
from typing import Any, AsyncIterator, List, Dict, Tuple
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
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt_old import GRAPH_FIELD_SEP, PROMPTS, get_prompt
import time


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
    prompt_template = PROMPTS["summarize_entity_descriptions"]
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
    # already_mapping = {}

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

        # already_mapping = {**already_node["already_mapping"]}

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
    description = GRAPH_FIELD_SEP.join(
        [dp["description"] for dp in nodes_data] + already_description
    )
    # source_id = GRAPH_FIELD_SEP.join(
    #     set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    # )

    source_id = GRAPH_FIELD_SEP.join(
        [dp["source_id"] for dp in nodes_data] + already_source_ids
    )

    # description = await _handle_entity_relation_summary(
    #     entity_name, description, global_config
    # )

    # for dp in nodes_data:
    #     already_mapping[dp["description"]] = dp["source_id"] 

    
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
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
    # keywords = GRAPH_FIELD_SEP.join(
    #     sorted(
    #         set(
    #             [dp["keywords"] for dp in edges_data if dp.get("keywords")]
    #             + already_keywords
    #         )
    #     )
    # )

    keywords = GRAPH_FIELD_SEP.join(
                [dp["keywords"] for dp in edges_data if dp.get("keywords")]
                + already_keywords
    )
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
                },
            )
    # description = await _handle_entity_relation_summary(
    #     f"({src_id}, {tgt_id})", description, global_config
    # )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


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

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    # entity_types = global_config["addon_params"].get(
    #     "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    # )
    _type = get_prompt("DEFAULT_ENTITY_TYPES", language)
    entity_types = global_config["addon_params"].get(
        "entity_types", _type
    )
    example_number = global_config["addon_params"].get("example_number", None)
    _examples = get_prompt("entity_extraction_examples", language)

    # if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
    #     examples = "\n".join(
    #         PROMPTS["entity_extraction_examples"][: int(example_number)]
    #     )
    # else:
    #     examples = "\n".join(PROMPTS["entity_extraction_examples"])

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

    # entity_extract_prompt = PROMPTS["entity_extraction"]
    entity_extract_prompt = get_prompt("entity_extraction", language)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    # continue_prompt = PROMPTS["entiti_continue_extraction"]
    # if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
    continue_prompt = get_prompt("entiti_continue_extraction", language)
    if_loop_prompt = get_prompt("entity_extraction", language)

    already_processed = 0
    already_entities = 0
    already_relations = 0

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
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await _user_llm_func_with_cache(hint_prompt)
        # final_result = await use_llm_func(hint_prompt)
        
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _user_llm_func_with_cache(
                continue_prompt, history_messages=history
            )
            # glean_result = await _user_llm_func_with_cache(
            #     continue_prompt, history_messages=history
            # )

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
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
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

    # print(maybe_nodes)
    # print(maybe_edges)
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

    if entity_vdb is not None:
    #     data_for_vdb = {}
    #     for dp in all_entities_data:
    #         list_des = list(dp["description"].split("<SEP>"))
    #         for des in list_des:
    #             data_for_vdb[compute_mdhash_id(dp["entity_name"] + des, prefix="ent-")] = {
    #                 "content": dp["entity_name"] + " " + des,
    #                 "entity_name": dp["entity_name"],
    #             }
        # await entities_vdb.upsert(data_for_vdb)
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
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
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst

def log_query_result(query_key, keywords, filename='results.jsonl'):
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump({query_key: keywords}, f, ensure_ascii=False)
        f.write('\n')

def is_query_cached(query_key, filename='results.jsonl'):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if query_key in record:
                    return record[query_key]
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return None

def is_query_cached(query_key, query_dict):
    return query_dict.get(query_key, None)

async def kg_retrieval(
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
    
    cache_keywords = is_query_cached(query,global_config["cache_queries"])
    # if cache_keywords:
    #     logger.info(f"Get cache for queries:{query}")
    #     hl_keywords, ll_keywords = cache_keywords["hl_keywords"], cache_keywords["ll_keywords"]
    # else:
    # Extract keywords using extract_keywords_only function which already supports conversation history
    if not query_param.use_query_for_retrieval:
        hl_keywords, ll_keywords = await extract_keywords_only(
            query, query_param, global_config, hashing_kv,cache_keywords
        )
        logger.debug(f"High-level keywords: {hl_keywords}")
        logger.debug(f"Low-level  keywords: {ll_keywords}")
    # Handle empty keywords
    # if hl_keywords == [] and ll_keywords == []:
    #     logger.warning("low_level_keywords and high_level_keywords is empty")
    #     return []
    # if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
    #     logger.warning(
    #         "low_level_keywords is empty, switching from %s mode to global mode",
    #         query_param.mode,
    #     )
    #     query_param.mode = "global"
    # if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
    #     logger.warning(
    #         "high_level_keywords is empty, switching from %s mode to local mode",
    #         query_param.mode,
    #     )
    #     query_param.mode = "local"

    if not query_param.use_query_for_retrieval:
        ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
        hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""
    else:
        ll_keywords_str, hl_keywords_str = query, query

    # Build context
    # chunk_list = await _build_retrieval_context(
    #     ll_keywords_str,
    #     hl_keywords_str,
    #     knowledge_graph_inst,
    #     entities_vdb,
    #     relationships_vdb,
    #     text_chunks_db,
    #     query_param,
    # )
    if query_param.ll_keyword_only:
        if ll_keywords_str: 
            chunk_list = await _build_retrieval_context(
                ll_keywords_str,
                ll_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )
        else:
            chunk_list = await _build_retrieval_context(
                hl_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )
    else:
        chunk_list = await _build_retrieval_context(
                ll_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
        )

    return chunk_list

async def _build_retrieval_context(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):        
    if not ll_keywords:
        ll_keywords = hl_keywords

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
    else:   # hybrid mode
        if query_param.retrieval_mode == "list_des": 
            ll_data, hl_data = await asyncio.gather(
                _get_node_data_list_des(
                    ll_keywords,
                    knowledge_graph_inst,
                    entities_vdb,
                    text_chunks_db,
                    query_param,
                ),
                _get_edge_data_list_des(
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
                ll_text_units_section_list,
                all_text_units_each_entity
            ) = ll_data

            (   use_entities,
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
                hl_text_units_section_list,
                text_unit_per_edges
            ) = hl_data

            ll_text_chunks = [(x[1],x[2],x[3],x[4]) for x in ll_text_units_section_list[1:]]
            hl_text_chunks = [(x[1],x[2],x[3],x[4]) for x in hl_text_units_section_list[1:]]
            # ll_text_chunks = [(x[1],x[2],x[3]) for x in ll_text_units_section_list[1:]]
            # hl_text_chunks = [(x[1],x[2],x[3]) for x in hl_text_units_section_list[1:]]
            # ll_text_chunks = [x[1] for x in ll_text_units_section_list[1:]]
            # hl_text_chunks = [x[1] for x in hl_text_units_section_list[1:]]
        elif query_param.retrieval_mode == "original":
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
                    ll_text_units_section_list,
                    all_text_units_each_entity
            ) = ll_data

            (   use_entities,
                    hl_entities_context,
                    hl_relations_context,
                    hl_text_units_context,
                    hl_text_units_section_list,
                    text_unit_per_edges
            ) = hl_data

            ll_text_chunks = [x[1] for x in ll_text_units_section_list[1:]]
            hl_text_chunks = [x[1] for x in hl_text_units_section_list[1:]]


    return ll_text_chunks,hl_text_chunks,all_text_units_each_entity,text_unit_per_edges


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
    
    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
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
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
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


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    cache_keywords: Dict = None
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
            hl_keywords, ll_keywords = keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
            if not cache_keywords:
                log_query_result(text, {"hl_keywords" : hl_keywords, "ll_keywords" : ll_keywords}, global_config["cache_queries_file"])
            return hl_keywords, ll_keywords
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )
    if cache_keywords:
        hl_keywords, ll_keywords = cache_keywords["hl_keywords"], cache_keywords["ll_keywords"]
        return hl_keywords, ll_keywords

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
    log_query_result(text, {"hl_keywords" : hl_keywords, "ll_keywords" : ll_keywords}, global_config["cache_queries_file"])
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
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return {"kg_context": kg_context, "vector_context": vector_context}

    # 5. Construct hybrid prompt
    sys_prompt = (
        system_prompt
        if system_prompt
        else PROMPTS["mix_rag_response"].format(
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
        entities_context, relations_context, text_units_context = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = await _get_edge_data(
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
        ) = ll_data

        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        ) = hl_data

        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    # not necessary to use LLM to generate a response
    if not entities_context.strip() and not relations_context.strip():
        return None

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

    # result_deduplicate = []
    # node_name = []
    # for result in results:
    #     if result["entity_name"] not in node_name:
    #         result_deduplicate.append(result)
    #         node_name.append(result["entity_name"])

    # results = result_deduplicate
    node_name = [r["entity_name"] for r in results]
    for n in node_name[:20]:
        logger.info(f"Extract nodes name:{n}")

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

    # node_name = [x["entity_name"] for x in node_datas]
    # for n in node_name:
    #     logger.info(f"Extract nodes name:{n}")
    (use_text_units,all_text_units_each_entity), use_relations = await asyncio.gather(
        _find_most_related_text_unit_from_entities(
            node_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
        _find_most_related_edges_from_entities(
            node_datas, query_param, knowledge_graph_inst
        ),
    )

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

    # import json 
    # with open("/home/hungpv/projects/TN/data/data_zalo_QA/filtered_corpus.json", "r") as f:
    #     corpus = json.load(f)

    # reverse_corpus = {v:k for k,v in corpus.items()}
    # chunk_id = [reverse_corpus[chunk] for chunk in use_text_units]
    # logger.info(f"Extract nodes:{chunk_id[:15]}")

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
    return entities_context, relations_context, text_units_context,text_units_section_list,all_text_units_each_entity

async def _get_node_data_list_des(
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
    
    if query_param.unique_entity_edge:
        print("Take uniques entities")
        top_k_candidates = 100
        result_chosen = []
        chunk_id_list = []
        node_name = []
        for result in results:
            if result["entity_name"] in node_name:
                continue
            if result["chunk_id"] not in node_name:
                chunk_id_list.append(result["chunk_id"])
                node_name.append(result["entity_name"])
                result_chosen.append(result)
            if len(chunk_id_list) >= top_k_candidates:
                print("Take top 100 uniques chunks")
                break

    else:
        top_k_candidates = 100
        result_chosen = []
        chunk_id_list = []
        for result in results:
            if result["chunk_id"] not in chunk_id_list:
                chunk_id_list.append(result["chunk_id"])
                result_chosen.append(result)
            if len(chunk_id_list) >= top_k_candidates:
                print("Take top 100 uniques chunks")
                break

    # results = result_deduplicate
    
    
    # # take the chunk id from the results
    distances = [float(r["distance"]) for r in result_chosen]
    chunk_ids = [dp["chunk_id"] for dp in result_chosen]
    description = [dp["description"] for dp in result_chosen]
    entity_name = [dp["entity_name"] for dp in result_chosen]
    chunk_datas = await text_chunks_db.get_by_ids(chunk_ids)
    use_text_units = [(chunk_data, distance,des,e) for chunk_data, distance, des,e in zip(chunk_datas, distances, description,entity_name)]

    # deduplicate the entity
    result_deduplicate = []
    node_name = []
    for result in results:
        if result["entity_name"] not in node_name:
            result_deduplicate.append(result)
            node_name.append(result["entity_name"])

    results = result_deduplicate

    for n in node_name[:20]:
        logger.info(f"Extract nodes name:{n}")

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
        logger.warning("Some nodprocessedes are missing, maybe the storage is damaged")

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk

    # (_,all_text_units_each_entity), use_relations = await asyncio.gather(
    #         _find_most_related_text_unit_from_entities(
    #             node_datas, query_param, text_chunks_db, knowledge_graph_inst
    #         ),
    #         _find_most_related_edges_from_entities(
    #             node_datas, query_param, knowledge_graph_inst
    #         ),
    #     )

    if query_param.retrieval_nodes:
        (_,all_text_units_each_entity), use_relations = await asyncio.gather(
            _find_most_related_text_unit_from_entities(
                node_datas, query_param, text_chunks_db, knowledge_graph_inst
            ),
            _find_most_related_edges_from_entities(
                node_datas, query_param, knowledge_graph_inst
            ),
        )
    else:
        all_text_units_each_entity = {}
        use_relations = await _find_most_related_edges_from_entities(
                node_datas, query_param, knowledge_graph_inst
            )

    len_node_datas = len(node_datas)
    # node_datas = truncate_list_by_token_size(
    #     node_datas,
    #     key=lambda x: x["description"],
    #     max_token_size=query_param.max_token_for_local_context,
    # )
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

    text_units_section_list = [["id", "content", "score"]]
    for i, (t,s,d,e) in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"], s,d,e])
    text_units_context = list_of_list_to_csv([[x[0], x[1]] for x in text_units_section_list])

    # text_units_section_list = [["id", "content", "score"]]
    # for i, (t,s,e) in enumerate(use_text_units):
    #     text_units_section_list.append([i, t["content"], s,e])
    # text_units_context = list_of_list_to_csv([[x[0], x[1]] for x in text_units_section_list])

    # text_units_section_list = [["id", "content"]]
    # for i, t in enumerate(use_text_units):
    #     text_units_section_list.append([i, t["content"]])
    # text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context,text_units_section_list,all_text_units_each_entity

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

    node_name = [x["entity_name"] for x in node_datas]
    # print(f"Node order 1: {node_name[:10]}")
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

    # all_text_units = truncate_list_by_token_size(
    #     all_text_units,
    #     key=lambda x: x["data"]["content"],
    #     max_token_size=query_param.max_token_for_text_unit,
    # )

    logger.info(
        f"Truncate chunks from {len(all_text_units_lookup)} to {len(all_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units_each_entity = {}
    # list_entity = []
    # for text_unit in all_text_units:
    #     entity_name = node_datas[text_unit["order"]]["entity_name"]
    #     if entity_name not in list_entity:
    #         list_entity.append(entity_name)
    #     all_text_units_each_entity[entity_name] = all_text_units_each_entity.get(entity_name, []) + [text_unit["data"]["content"]]
    
    # print(f"Node oder 2: {list_entity[:10]}")

    for i in range(len(node_name)):
        results_text_chunks = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id in text_units[i]]
    )
        all_text_units_each_entity[node_name[i]] = all_text_units_each_entity.get(node_name[i], []) + [text_chunk["content"] for text_chunk in results_text_chunks]

    # for n in node_name[:20]:
    #     logger.info(f"Extract nodes name:{n} - # Chunks: {len(all_text_units_each_entity[n])}")

    total_chunks = 0
    for k , v in all_text_units_each_entity.items(): 
        total_chunks += len(v)

    print(total_chunks) 

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units, all_text_units_each_entity


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

    for edge_data in edge_datas[:15]:
        logger.info(f"<{edge_data['src_id']},{edge_data['tgt_id']}>")


    # edge_datas = sorted(
    #     edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    # )
    len_edge_datas = len(edge_datas)
    # edge_datas = truncate_list_by_token_size(
    #     edge_datas,
    #     key=lambda x: x["description"],
    #     max_token_size=query_param.max_token_for_global_context,
    # )
    logger.debug(
        f"Truncate relations from {len_edge_datas} to {len(edge_datas)} (max tokens:{query_param.max_token_for_global_context})"
    )

    use_entities, (use_text_units,text_unit_per_edges) = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
    )

    
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    # import json 
    # with open("/home/hungpv/projects/TN/data/data_zalo_QA/filtered_corpus.json", "r") as f:
    #     corpus = json.load(f)

    # reverse_corpus = {v:k for k,v in corpus.items()}
    # chunk_id = [reverse_corpus[chunk["cotent"]] for chunk in use_text_units]
    # logger.info(f"Extract nodes:{chunk_id[:20]}")

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
    return use_entities, entities_context, relations_context, text_units_context,text_units_section_list,text_unit_per_edges

async def _get_edge_data_list_des(
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

    # take the chunk id from the results

    if query_param.unique_entity_edge:
    #################################### unique edge ##################################
        top_k_candidates = 100
        result_chosen = []
        chunk_id_list = []
        edge_name = []
        for result in results:
            if (result["src_id"], result["tgt_id"]) in edge_name:
                continue
            if result["chunk_id"] not in chunk_id_list:
                result_chosen.append(result)
                edge_name.append((result["src_id"], result["tgt_id"]))
                chunk_id_list.append(result["chunk_id"])

            if len(chunk_id_list) >= top_k_candidates:
                print("Take top 100 uniques chunks")
                break

    else:
        #################################### non-unique edge ##################################
        top_k_candidates = 100
        result_chosen = []
        chunk_id_list = []
        for result in results:
            if result["chunk_id"] not in chunk_id_list:
                result_chosen.append(result)
                chunk_id_list.append(result["chunk_id"])

            if len(chunk_id_list) >= top_k_candidates:
                print("Take top 100 uniques chunks")
                break


    # take the chunk id from the results
    distances = [float(r["distance"]) for r in result_chosen]
    chunk_ids = [dp["chunk_id"] for dp in result_chosen]
    description = [dp["description"] for dp in result_chosen]
    edge_name_text = [" <|> ".join([dp["src_id"], dp["tgt_id"]]) for dp in result_chosen] 
    chunk_datas = await text_chunks_db.get_by_ids(chunk_ids)
    use_text_units = [(chunk_data, distance, des,e) for chunk_data, distance,des,e in zip(chunk_datas, distances, description,edge_name_text)]

    
    # Filter những edges bị trùng
    result_deduplicate = []
    edge_name = []
    for result in results:
        if (result["src_id"], result["tgt_id"]) not in edge_name:
            result_deduplicate.append(result)
            edge_name.append((result["src_id"], result["tgt_id"]))

    results = result_deduplicate


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

    # for r in results[:15]:
    #     logger.info(f"<{r['src_id']},{r['tgt_id']}>")

    for src, tgt in edge_name[:20]:
        logger.info(f"<{src},{tgt}>")


    # edge_datas = sorted(
    #     edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    # )
    len_edge_datas = len(edge_datas)
    # edge_datas = truncate_list_by_token_size(
    #     edge_datas,
    #     key=lambda x: x["description"],
    #     max_token_size=query_param.max_token_for_global_context,
    # )
    logger.debug(
        f"Truncate relations from {len_edge_datas} to {len(edge_datas)} (max tokens:{query_param.max_token_for_global_context})"
    )

    if query_param.retrieval_nodes:
        use_entities, (_,text_unit_per_edges) = await asyncio.gather(
            _find_most_related_entities_from_relationships(
                edge_datas, query_param, knowledge_graph_inst
            ),
            _find_related_text_unit_from_relationships(
                edge_datas, query_param, text_chunks_db, knowledge_graph_inst
            ),
        )
    else:
        text_unit_per_edges = {}
        use_entities = await _find_most_related_entities_from_relationships(
                edge_datas, query_param, knowledge_graph_inst
            )

    # use_entities, (use_text_units,text_unit_per_edges) = await asyncio.gather(
    #     _find_most_related_entities_from_relationships(
    #         edge_datas, query_param, knowledge_graph_inst
    #     ),
    #     _find_related_text_unit_from_relationships(
    #         edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    #     ),
    # )

    
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

    text_units_section_list = [["id", "content", "score"]]
    for i, (t,s,d,e) in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"],s,d,e])

    # text_units_section_list = [["id", "content", "score"]]
    # for i, (t,s,e) in enumerate(use_text_units):
    #     text_units_section_list.append([i, t["content"],s,e])
    # text_units_section_list = [["id", "content"]]
    # for i, t in enumerate(use_text_units):
    #     text_units_section_list.append([i, t["content"]])
 # text_units_section_list = [["id", "content", "score"]]
    # for i, (t,s,e) in enumerate(use_text_units):
    #     text_units_sectio
    text_units_context = list_of_list_to_csv([[x[0], x[1]] for x in text_units_section_list])
    return use_entities, entities_context, relations_context, text_units_context,text_units_section_list,text_unit_per_edges

async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()
    # point = len(edge_datas) + 1

    for e in edge_datas:
        if e["src_id"] not in seen:
            # entity_names[e["src_id"]] = point
            seen.add(e["src_id"])
            entity_names.append(e["src_id"])
        # else:
            # entity_names[e["src_id"]] += point
        if e["tgt_id"] not in seen:
            # entity_names[e["tgt_id"]] = point
            seen.add(e["tgt_id"])
            entity_names.append(e["tgt_id"])
        # else:
            # entity_names[e["tgt_id"]] += point
        
        # point -= 1

    # entity_names = list(entity_names.items())
    # entity_names = sorted(entity_names, key = lambda x : x[1], reverse = True)

    # entity_names = [x[0] for x in entity_names]

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
    # node_datas = truncate_list_by_token_size(
    #     node_datas,
    #     key=lambda x: x["description"],
    #     max_token_size=query_param.max_token_for_local_context,
    # )
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

    edge_name = [edge_data["src_id"] + " <-> " + edge_data["tgt_id"] for edge_data in edge_datas]

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

    # truncated_text_units = truncate_list_by_token_size(
    #     valid_text_units,
    #     key=lambda x: x["data"]["content"],
    #     max_token_size=query_param.max_token_for_text_unit,
    # )
    truncated_text_units = valid_text_units

    text_unit_per_edges = {}
    # for t in truncated_text_units:
    #     edge_data = edge_datas[t["order"]]
    #     edge_name = edge_data["src_id"] + " <-> " + edge_data["tgt_id"]
    #     text_unit_per_edges[edge_name] = text_unit_per_edges.get(edge_name, []) + [t["data"]["content"]]

    for i in range(len(edge_name)):
        results_text_chunks = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id in text_units[i]]
    )
        text_unit_per_edges[edge_name[i]] = text_unit_per_edges.get(edge_name[i], []) + [text_chunk["content"] for text_chunk in results_text_chunks]


    logger.debug(
        f"Truncate chunks from {len(valid_text_units)} to {len(truncated_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units,text_unit_per_edges


# async def get_chunk_ids_from_entity_or_edge(
#     item_id: str | tuple[str, str],
#     knowledge_graph_inst: BaseGraphStorage
# ):
#     """
#     Retrieve all chunk IDs associated with an entity node or an edge.
    
#     Args:
#         item_id: Either a string (entity node ID) or a tuple of two strings (source and target IDs for an edge)
#         knowledge_graph_inst: Instance of BaseGraphStorage for accessing the knowledge graph
#         text_chunks_db: Instance of BaseKVStorage for accessing text chunks
        
#     Returns:
#         list[str]: List of chunk IDs associated with the entity or edge
#     """
#     if isinstance(item_id, tuple) and len(item_id) == 2:
#         # Handle edge case - get chunks from the edge
#         src_id, tgt_id = item_id
#         edge_data = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        
#         if edge_data is None or "source_id" not in edge_data:
#             logger.warning(f"Edge <{src_id},{tgt_id}> not found or doesn't have source_id")
#             return []
            
#         chunk_ids = split_string_by_multi_markers(edge_data["source_id"], [GRAPH_FIELD_SEP])
#         return chunk_ids
        
#     else:
#         # Handle entity node case
#         node_data = await knowledge_graph_inst.get_node(item_id)
        
#         if node_data is None or "source_id" not in node_data:
#             logger.warning(f"Entity node {item_id} not found or doesn't have source_id")
#             return []
            
#         chunk_ids = split_string_by_multi_markers(node_data["source_id"], [GRAPH_FIELD_SEP])
#         return chunk_ids


# # Lấy chunk ID từ một entity node
# chunk_ids = await get_chunk_ids_from_entity_or_edge("entity_name", knowledge_graph_inst, text_chunks_db)

# # Lấy chunk ID từ một edge
# chunk_ids = await get_chunk_ids_from_entity_or_edge(("source_id", "target_id"), knowledge_graph_inst, text_chunks_db)


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

async def naive_retrieval(query: str,
                          query_param: QueryParam,
                          global_config: dict[str, str],
                          chunks_vdb: BaseVectorStorage,
                          text_chunks_db: BaseKVStorage):

    # language = global_config["addon_params"].get(
    #     "language", PROMPTS["DEFAULT_LANGUAGE"]
    # )
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    # if not len(results):
    #     return get_prompt("fail_response", language)

    chunks_ids = [r["id"] for r in results]
    chunks_distance = [float(r["distance"]) for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    valid_chunks = [chunk["content"] for chunk in valid_chunks]

    # if not valid_chunks:
    #     logger.warning("No valid chunks found after filtering")
    #     return get_prompt("fail_response", language)
    chunk_scores_dict = {}
    for i in range(len(valid_chunks)):
        chunk_scores_dict[valid_chunks[i]] = chunks_distance[i]

    return chunk_scores_dict

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

    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

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

    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
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
        return PROMPTS["fail_response"]
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
        return PROMPTS["fail_response"]

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

    sys_prompt_temp = PROMPTS["rag_response"]
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



async def retrieve_node_details_for_recall(
    query: str,
    entities_vdb: BaseVectorStorage,
    knowledge_graph_inst: BaseGraphStorage,
    # text_chunks_db: BaseKVStorage, # Tùy chọn: Nếu bạn muốn lấy cả nội dung chunk gốc
    query_param: QueryParam,
) -> list[dict]:
    """
    Truy vấn vector store của thực thể (entities_vdb) dựa trên query,
    sau đó lấy mô tả và *tất cả* chunk ID gốc của node từ knowledge graph.

    Hàm này trả về danh sách các dictionary, mỗi dict đại diện cho một *kết quả truy vấn VDB*,
    chứa thông tin về thực thể liên quan, điểm số, chunk ID gây ra truy vấn,
    và danh sách tất cả chunk ID của thực thể đó trong KG.

    Args:
        query: Chuỗi truy vấn đầu vào.
        entities_vdb: Instance của BaseVectorStorage cho thực thể.
        knowledge_graph_inst: Instance của BaseGraphStorage.
        # text_chunks_db: (Tùy chọn) Instance BaseKVStorage nếu muốn lấy nội dung text chunk.
        query_param: Instance của QueryParam, chủ yếu để lấy top_k.

    Returns:
        Một list các dictionary, mỗi dict chứa:
        - 'entity_name' (str): Tên của thực thể liên quan đến VDB hit.
        - 'kg_description' (str | None): Mô tả của thực thể từ knowledge graph.
        - 'retrieved_chunk_id' (str): ID của chunk cụ thể từ VDB hit.
        - 'score' (float): Điểm tương đồng (thường là distance) từ VDB hit.
        - 'all_kg_chunk_ids' (list[str]): Danh sách tất cả chunk ID của thực thể này trong KG.
        # - 'all_kg_chunks_content' (list[str] | None): (Tùy chọn) Nội dung của các chunk gốc.
        - 'entity_appearance_count' (int | None): Số lần entity_name xuất hiện trong KQ VDB ban đầu (chỉ có khi unique_entity_edge=True).
        - 'triggering_chunk_appearance_count' (int | None): Số lần retrieved_chunk_id xuất hiện trong KQ VDB ban đầu (chỉ có khi unique_entity_edge=True).
    """
    logger.info(
        f"Retrieving node details for recall: '{query}', top_k: {query_param.top_k}"
    )

    # 1. Truy vấn Vector Store của Thực thể (entities_vdb)
    try:
        vdb_results = await entities_vdb.query(query, top_k=query_param.top_k)
        print("oke vdb khong")
        # print(vdb_results)
    except Exception as e:
        logger.error(f"Error querying entities_vdb: {e}")
        return []

    if not vdb_results:
        logger.warning("No results found in entities_vdb for query.")
        return []

    # --- Revised Step 2: Filter VDB results and Count Appearances during filtering ---
    from collections import Counter # Ensure Counter is imported if not already
    entity_counts = Counter()
    chunk_counts = Counter()
    result_chosen = []
    top_k_candidates = 300 # Limit for candidates after filtering
    # print("den day chua")
    if getattr(query_param, 'unique_entity_edge', False):
        print("unique_entity_edge is True")
        logger.debug("Filtering for unique entities AND unique triggering chunks, counting appearances until limit.")
        seen_entities = set()
        seen_triggering_chunks = set() # Thêm set để theo dõi chunk đã dùng để trigger
        for result in vdb_results: # Iterate through original VDB results
            entity_name = result.get("entity_name")
            chunk_id = result.get("chunk_id")
            distance = result.get("distance")
            
            # Basic validation before counting or processing
            if entity_name and chunk_id is not None and distance is not None:
                try:
                    float(distance) # Validate distance format
                except (ValueError, TypeError):
                    logger.warning(f"Skipping result due to invalid distance: {result}")
                    continue # Skip this result entirely

                # Increment counts regardless of whether it's chosen
                entity_counts[entity_name] += 1
                chunk_counts[chunk_id] += 1

                # Check for uniqueness of BOTH entity and triggering chunk
                if entity_name not in seen_entities and chunk_id not in seen_triggering_chunks:
                    seen_entities.add(entity_name)
                    seen_triggering_chunks.add(chunk_id) # Đánh dấu chunk này đã được dùng để trigger

                    # Get current counts accumulated so far
                    current_entity_count = entity_counts[entity_name]
                    current_chunk_count = chunk_counts[chunk_id]

                    result_chosen.append({
                        **result,
                        "entity_appearance_count": current_entity_count,
                        "chunk_appearance_count": current_chunk_count
                    })

                    # Check if we have enough candidates AFTER adding
                    if len(result_chosen) >= top_k_candidates:
                        logger.debug(f"Reached top {top_k_candidates} unique (entity, triggering_chunk) pairs. Stopping filtering.")
                        break # Stop processing further vdb_results
                # else: Bỏ qua nếu entity đã thấy HOẶC chunk trigger đã thấy
                elif entity_name not in seen_entities and chunk_id in seen_triggering_chunks:
                     logger.debug(f"Skipping entity '{entity_name}' because triggering chunk '{chunk_id}' was already used.")
                # Implicitly skips if entity_name in seen_entities

            # else: skip results missing entity/chunk_id or distance

    else: # Filter for unique chunks
        logger.debug("Filtering for unique chunks, counting appearances until limit.")
        print("unique_entity_edge is False")
        seen_chunk_ids = set()
        for result in vdb_results: # Iterate through original VDB results
            entity_name = result.get("entity_name")
            chunk_id = result.get("chunk_id", "UNKONWN_CHUNK_ID") # Default to UNKNOWN if not present
            distance = result.get("distance")
            # print("chet o sau day")
            # Basic validation before counting or processing
            if entity_name and chunk_id is not None and distance is not None:
                try:
                    float(distance) # Validate distance format
                except (ValueError, TypeError):
                    logger.warning(f"Skipping result due to invalid distance: {result}")
                    continue # Skip this result entirely
                # print("hay o day")
                # Increment counts for the valid result being processed
                entity_counts[entity_name] += 1
                chunk_counts[chunk_id] += 1

                # Check for uniqueness *after* counting
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    # Get current counts accumulated so far
                    current_entity_count = entity_counts[entity_name] # Count for the entity associated with this chunk
                    current_chunk_count = chunk_counts[chunk_id]

                    result_chosen.append({
                        **result,
                        "entity_appearance_count": current_entity_count,
                        "chunk_appearance_count": current_chunk_count
                    })

                    # Check if we have enough candidates AFTER adding
                    if len(result_chosen) >= top_k_candidates:
                        logger.debug(f"Reached top {top_k_candidates} unique chunks. Stopping filtering.")
                        break # Stop processing further vdb_results
            # else: skip results missing entity/chunk_id or distance

    if not result_chosen:
        logger.warning("No results left after filtering VDB hits based on mode.")
        return []

    logger.info(f"Filtered VDB results down to {len(result_chosen)} candidates based on mode, with counts accumulated during filtering.")

    # --- End of Revised Step 2 ---

    # Steps 3 and 4 (KG fetch and final assembly) remain the same
    # as they already expect the counts in `result_chosen`.

    # 3. Chuẩn bị lấy thông tin từ Knowledge Graph (từ kết quả đã lọc)
    node_names_to_fetch = set()
    processed_vdb_results = []
    for result in result_chosen: # Sử dụng kết quả đã lọc (giờ luôn có counts)
        entity_name = result.get("entity_name")
        chunk_id = result.get("chunk_id")
        score = result.get("distance")
        vdb_hit_description = result.get("description", "UNKNOWN_VDB_DESCRIPTION") # Lấy description từ VDB hit
        # Lấy appearance counts (giờ luôn có giá trị từ bước trước)
        entity_appearance_count = result.get("entity_appearance_count")
        chunk_appearance_count = result.get("chunk_appearance_count") # Đã đổi tên

        # Kiểm tra lại lần nữa (mặc dù đã lọc ở trên) và chuyển đổi score
        if entity_name and chunk_id is not None and score is not None:
            try:
                float_score = float(score)
                node_names_to_fetch.add(entity_name)
                processed_vdb_results.append({
                    "entity_name": entity_name,
                    "retrieved_chunk_id": chunk_id,
                    "score": float_score,
                    "vdb_hit_description": vdb_hit_description, # Thêm VDB description vào đây
                    "entity_appearance_count": entity_appearance_count,
                    "chunk_appearance_count": chunk_appearance_count
                })
            except (ValueError, TypeError) as e:
                 logger.warning(f"Could not convert score '{score}' to float for filtered entity '{entity_name}', chunk '{chunk_id}'. Skipping. Error: {e}")
        # else: Lỗi này không nên xảy ra nếu pre-validation hoạt động đúng

    if not node_names_to_fetch:
        logger.warning("No valid entity names found in filtered VDB results to fetch from KG.")
        return []

    # Lấy thông tin node từ KG bất đồng bộ
    node_fetch_tasks = [knowledge_graph_inst.get_node(name) for name in node_names_to_fetch]
    try:
        fetched_node_data_list = await asyncio.gather(*node_fetch_tasks)
    except Exception as e:
        logger.error(f"Error fetching node data from knowledge graph: {e}")
        fetched_node_data_list = [None] * len(node_names_to_fetch)

    # Tạo map từ tên node sang thông tin KG (description và chunk IDs)
    kg_node_info_map = {}
    for name, node_data in zip(node_names_to_fetch, fetched_node_data_list):
        if node_data:
            # Lấy description tổng hợp từ KG node
            kg_node_description = node_data.get("description", "UNKNOWN_KG_DESCRIPTION")
            source_id_str = node_data.get("source_id", "")
            all_chunk_ids = split_string_by_multi_markers(source_id_str, [GRAPH_FIELD_SEP])
            # Filter out empty strings that might result from splitting
            all_chunk_ids = [cid for cid in all_chunk_ids if cid]
            kg_node_info_map[name] = {
                # "kg_description": kg_node_description, # Description từ KG
                "all_kg_chunk_ids": all_chunk_ids
            }
        else:
            # Handle case where node wasn't found in KG
            kg_node_info_map[name] = {
                # "kg_description": None, # Không có description từ KG
                "all_kg_chunk_ids": []
            }
            logger.debug(f"Node '{name}' not found in KG or had no data.")

    # 4. Kết hợp thông tin từ VDB (đã lọc và có counts) và KG
    final_output = []
    for vdb_result in processed_vdb_results: # Lặp qua kết quả VDB đã xử lý
        entity_name = vdb_result["entity_name"]
        kg_info = kg_node_info_map.get(entity_name) # Lấy thông tin KG đã fetch

        if kg_info: # Chỉ thêm kết quả nếu tìm thấy thông tin KG tương ứng
            final_output.append({
                "entity_name": entity_name,
                # "kg_description": kg_info["kg_description"], # Description tổng hợp từ KG node
                "description": vdb_result["vdb_hit_description"], # Description cụ thể từ VDB hit
                "retrieved_chunk_id": vdb_result["retrieved_chunk_id"],
                "score": vdb_result["score"],
                "all_kg_chunk_ids": kg_info["all_kg_chunk_ids"],
                "entity_appearance_count": vdb_result["entity_appearance_count"],
                "chunk_appearance_count": vdb_result["chunk_appearance_count"]
            })
        # else: Bỏ qua VDB result nếu không tìm thấy node tương ứng trong KG (đã log ở trên)

    # (Phần lấy nội dung chunk tùy chọn giữ nguyên)

    # Sắp xếp kết quả cuối cùng theo score (ví dụ: distance thấp hơn là tốt hơn)
    final_output.sort(key=lambda x: x["score"])

    logger.info(f"Returning {len(final_output)} node detail results for recall.")
    return final_output

# --- Cách sử dụng ví dụ ---
# async def main():
#     # Giả định bạn đã khởi tạo các instance cần thiết:
#     # your_entities_vdb_instance: BaseVectorStorage
#     # your_kg_instance: BaseGraphStorage
#     # your_query_param_instance = QueryParam(top_k=5)
#     # your_text_chunks_db_instance: BaseKVStorage (nếu muốn lấy content)

#     results = await retrieve_node_details_for_recall(
#         query="Giá của dịch vụ ABC là bao nhiêu?",
#         entities_vdb=your_entities_vdb_instance,
#         knowledge_graph_inst=your_kg_instance,
#         # text_chunks_db=your_text_chunks_db_instance, # Bỏ comment nếu dùng
#         query_param=your_query_param_instance
#     )

#     for res in results:
#         print("-" * 20)
#         print(f"Entity Name: {res['entity_name']}")
#         print(f"KG Description: {res['kg_description']}")
#         print(f"Retrieved Chunk ID: {res['retrieved_chunk_id']}")
#         print(f"Score: {res['score']:.4f}")
#         print(f"All KG Chunk IDs ({len(res['all_kg_chunk_ids'])}): {res['all_kg_chunk_ids'][:5]}...") # In 5 cái đầu
#         # if 'all_kg_chunks_content' in res:
#         #     print(f"All KG Chunks Content ({len(res['all_kg_chunks_content'])}):")
#         #     for content in res['all_kg_chunks_content'][:2]: # In 2 nội dung đầu
#         #         print(f"  - {content[:80]}...") # In 80 ký tự đầu

# if __name__ == "__main__":
#     # Nhớ import asyncio và các class cần thiết
#     # from lightrag.base import QueryParam
#     # from lightrag.prompt_old import GRAPH_FIELD_SEP
#     # from lightrag.utils import split_string_by_multi_markers, logger
#     # ... giả định các import khác và khởi tạo instance ...
#     # asyncio.run(main())
#     pass
# Thêm hàm này vào file lightrag/operate_old.py
# (Có thể đặt sau hàm retrieve_node_details_for_recall)

from collections import Counter # Đảm bảo đã import ở đầu file hoặc trong hàm

async def retrieve_edge_details_for_recall(
    keywords: str, # Thay query bằng keywords cho ngữ nghĩa cạnh
    relationships_vdb: BaseVectorStorage,
    knowledge_graph_inst: BaseGraphStorage,
    text_chunks_db: BaseKVStorage, # Tùy chọn lấy nội dung chunk
    query_param: QueryParam,
) -> list[dict]:
    """
    Truy vấn vector store của quan hệ (relationships_vdb) dựa trên keywords,
    sau đó lấy mô tả cạnh từ KG, mô tả từ VDB hit, và *tất cả* chunk ID gốc của cạnh từ KG.
    Đồng thời đếm số lần xuất hiện của cạnh và chunk trong quá trình lọc.

    Args:
        keywords: Chuỗi keywords để truy vấn VDB quan hệ.
        relationships_vdb: Instance BaseVectorStorage cho quan hệ.
        knowledge_graph_inst: Instance BaseGraphStorage.
        text_chunks_db: (Tùy chọn) Instance BaseKVStorage nếu muốn lấy nội dung text chunk.
        query_param: Instance QueryParam, dùng cho top_k và unique_entity_edge.

    Returns:
        Một list các dictionary, mỗi dict chứa:
        - 'src_id' (str): ID nút nguồn của cạnh.
        - 'tgt_id' (str): ID nút đích của cạnh.
        - 'kg_description' (str | None): Mô tả của cạnh lấy từ knowledge graph.
        - 'vdb_hit_description' (str): Description cụ thể từ VDB hit đã khớp query.
        - 'retrieved_chunk_id' (str): ID của chunk cụ thể từ VDB hit liên quan đến cạnh.
        - 'score' (float): Điểm tương đồng (distance) từ VDB hit.
        - 'all_kg_chunk_ids' (list[str]): Danh sách tất cả chunk ID của cạnh này trong KG.
        - 'edge_appearance_count' (int): Số lần cạnh (src, tgt) xuất hiện trong KQ VDB được xử lý cho đến khi đủ candidates.
        - 'chunk_appearance_count' (int): Số lần retrieved_chunk_id xuất hiện trong KQ VDB được xử lý cho đến khi đủ candidates.
        # - 'all_kg_chunks_content' (list[str] | None): (Tùy chọn) Nội dung của các chunk gốc.
        - (Có thể thêm các trường khác từ KG edge data nếu cần, ví dụ: weight, keywords...)
    """
    logger.info(
        f"Retrieving edge details for recall: keywords='{keywords}', top_k={query_param.top_k}, unique_mode={query_param.unique_entity_edge}"
    )

    # 1. Truy vấn Vector Store của Quan hệ (relationships_vdb)
    try:
        # Sử dụng top_k trực tiếp như hàm _get_edge_data_list_des cũ
        # Có thể cân nhắc tăng lên nếu cần lọc nhiều hơn, ví dụ query_param.top_k * 5
        vdb_results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
    except Exception as e:
        logger.error(f"Error querying relationships_vdb: {e}")
        return []

    if not vdb_results:
        logger.warning("No results found in relationships_vdb for keywords.")
        return []

    # --- Step 2: Filter VDB results and Count Appearances during filtering ---
    edge_counts = Counter() # Đếm (src, tgt)
    chunk_counts = Counter()
    result_chosen = []
    top_k_candidates = 300 # Giới hạn số lượng kết quả sau khi lọc

    if getattr(query_param, 'unique_entity_edge', False): # True nghĩa là unique edge
        logger.debug("Filtering for unique edges AND unique triggering chunks, counting appearances until limit.")
        seen_edges = set() # Lưu các tuple (src_id, tgt_id) đã thấy
        seen_triggering_chunks = set() # Thêm set cho chunk trigger
        for result in vdb_results:
            src_id = result.get("src_id")
            tgt_id = result.get("tgt_id")
            chunk_id = result.get("chunk_id")
            distance = result.get("distance")

            # Validation cơ bản
            if src_id and tgt_id and chunk_id is not None and distance is not None:
                try:
                    float(distance) # Validate distance
                except (ValueError, TypeError):
                    logger.warning(f"Skipping edge result due to invalid distance: {result}")
                    continue

                edge_key = tuple(sorted((src_id, tgt_id))) # Chuẩn hóa key cạnh

                # Tăng count *trước khi* kiểm tra unique
                edge_counts[edge_key] += 1
                chunk_counts[chunk_id] += 1

                # Kiểm tra unique của CẢ edge VÀ chunk trigger
                if edge_key not in seen_edges and chunk_id not in seen_triggering_chunks:
                    seen_edges.add(edge_key)
                    seen_triggering_chunks.add(chunk_id) # Đánh dấu chunk trigger đã dùng

                    # Lấy số đếm hiện tại
                    current_edge_count = edge_counts[edge_key]
                    current_chunk_count = chunk_counts[chunk_id]

                    result_chosen.append({
                        **result,
                        "edge_appearance_count": current_edge_count,
                        "chunk_appearance_count": current_chunk_count
                    })

                    if len(result_chosen) >= top_k_candidates:
                        logger.debug(f"Reached top {top_k_candidates} unique (edge, triggering_chunk) pairs. Stopping filtering.")
                        break
                 # else: Bỏ qua nếu edge đã thấy HOẶC chunk trigger đã thấy
                elif edge_key not in seen_edges and chunk_id in seen_triggering_chunks:
                     logger.debug(f"Skipping edge {edge_key} because triggering chunk '{chunk_id}' was already used.")
                # Implicitly skips if edge_key in seen_edges

            # else: skip results missing fields

    else: # False nghĩa là unique chunk
        logger.debug("Filtering for unique chunks (associated with edges), counting appearances until limit.")
        seen_chunk_ids = set()
        for result in vdb_results:
            src_id = result.get("src_id")
            tgt_id = result.get("tgt_id")
            chunk_id = result.get("chunk_id")
            distance = result.get("distance")

            # Validation cơ bản
            if src_id and tgt_id and chunk_id is not None and distance is not None:
                try:
                    float(distance) # Validate distance
                except (ValueError, TypeError):
                    logger.warning(f"Skipping edge result due to invalid distance: {result}")
                    continue

                edge_key = tuple(sorted((src_id, tgt_id))) # Chuẩn hóa key cạnh

                # Tăng count *trước khi* kiểm tra unique
                edge_counts[edge_key] += 1
                chunk_counts[chunk_id] += 1

                # Kiểm tra unique chunk
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                     # Lấy số đếm hiện tại
                    current_edge_count = edge_counts[edge_key] # Count cho cạnh liên quan đến chunk này
                    current_chunk_count = chunk_counts[chunk_id]

                    result_chosen.append({
                        **result,
                        "edge_appearance_count": current_edge_count,
                        "chunk_appearance_count": current_chunk_count
                    })

                    if len(result_chosen) >= top_k_candidates:
                        logger.debug(f"Reached top {top_k_candidates} unique chunks. Stopping filtering.")
                        break
            # else: skip results missing fields

    if not result_chosen:
        logger.warning("No results left after filtering relationship VDB hits based on mode.")
        return []

    logger.info(f"Filtered relationship VDB results down to {len(result_chosen)} candidates based on mode, with counts accumulated during filtering.")
    # --- End of Step 2 ---


    # 3. Chuẩn bị lấy thông tin từ Knowledge Graph (từ kết quả đã lọc)
    edge_keys_to_fetch = set() # Set các tuple (src_id, tgt_id)
    processed_vdb_results = []
    for result in result_chosen: # Giờ luôn có counts
        src_id = result.get("src_id")
        tgt_id = result.get("tgt_id")
        chunk_id = result.get("chunk_id")
        score = result.get("distance")
        vdb_hit_description = result.get("description", "UNKNOWN_VDB_DESCRIPTION") # Lấy desc từ VDB
        edge_appearance_count = result.get("edge_appearance_count")
        chunk_appearance_count = result.get("chunk_appearance_count")

        # Check lại các trường chính và score
        if src_id and tgt_id and chunk_id is not None and score is not None:
            try:
                float_score = float(score)
                edge_key = tuple(sorted((src_id, tgt_id))) # Dùng key đã chuẩn hóa
                edge_keys_to_fetch.add(edge_key) # Thêm key chuẩn hóa vào set để fetch
                processed_vdb_results.append({
                    "src_id": src_id, # Lưu ID gốc
                    "tgt_id": tgt_id, # Lưu ID gốc
                    "retrieved_chunk_id": chunk_id,
                    "score": float_score,
                    "vdb_hit_description": vdb_hit_description,
                    "edge_appearance_count": edge_appearance_count,
                    "chunk_appearance_count": chunk_appearance_count
                })
            except (ValueError, TypeError) as e:
                 logger.warning(f"Could not convert score '{score}' for edge ('{src_id}', '{tgt_id}'), chunk '{chunk_id}'. Skipping. Error: {e}")
        # else: Lỗi này ít xảy ra do đã check ở bước 2

    if not edge_keys_to_fetch:
        logger.warning("No valid edge keys found in filtered VDB results to fetch from KG.")
        return []

    # Lấy thông tin cạnh từ KG bất đồng bộ
    # Lưu ý: get_edge thường nhận (src, tgt) theo thứ tự, nên fetch bằng key gốc từ processed_vdb_results thì tốt hơn
    # Nhưng để tránh fetch trùng lặp, ta dùng edge_keys_to_fetch rồi map lại
    logger.debug(f"Fetching {len(edge_keys_to_fetch)} unique edges from KG.")
    edge_fetch_tasks = [knowledge_graph_inst.get_edge(src, tgt) for src, tgt in edge_keys_to_fetch]
    try:
        fetched_edge_data_list = await asyncio.gather(*edge_fetch_tasks)
    except Exception as e:
        logger.error(f"Error fetching edge data from knowledge graph: {e}")
        # Xử lý lỗi, ví dụ trả về list rỗng hoặc gán None cho tất cả
        return [] # Hoặc xử lý phức tạp hơn nếu cần

    # Tạo map từ key cạnh (đã chuẩn hóa) sang thông tin KG
    kg_edge_info_map = {}
    for edge_key, edge_data in zip(edge_keys_to_fetch, fetched_edge_data_list):
        if edge_data:
            kg_description = edge_data.get("description", "UNKNOWN_KG_DESCRIPTION")
            source_id_str = edge_data.get("source_id", "")
            all_chunk_ids = split_string_by_multi_markers(source_id_str, [GRAPH_FIELD_SEP])
            all_chunk_ids = [cid for cid in all_chunk_ids if cid] # Lọc chuỗi rỗng
            # Lấy thêm các trường khác nếu muốn (weight, keywords...)
            kg_weight = edge_data.get("weight")
            kg_keywords = edge_data.get("keywords")
            kg_edge_info_map[edge_key] = {
                "kg_description": kg_description,
                "all_kg_chunk_ids": all_chunk_ids,
                "kg_weight": kg_weight,
                "kg_keywords": kg_keywords
            }
        else:
            # Cạnh không tìm thấy trong KG
             kg_edge_info_map[edge_key] = {
                "kg_description": None,
                "all_kg_chunk_ids": [],
                "kg_weight": None,
                "kg_keywords": None
            }
            # logger.info(f"Edge with key {edge_key} not found in KG or had no data.")


    # 4. Kết hợp thông tin từ VDB (đã lọc) và KG
    final_output = []
    for vdb_result in processed_vdb_results:
        src_id = vdb_result["src_id"]
        tgt_id = vdb_result["tgt_id"]
        edge_key = tuple(sorted((src_id, tgt_id))) # Key chuẩn hóa để tra cứu
        kg_info = kg_edge_info_map.get(edge_key)

        if kg_info and kg_info["kg_description"] is not None: # Chỉ thêm nếu cạnh tồn tại trong KG
            final_output.append({
                "src_id": src_id,
                "tgt_id": tgt_id,
                # "kg_description": kg_info["kg_description"], # Desc từ KG
                "description": vdb_result["vdb_hit_description"], # Desc từ VDB hit
                "retrieved_chunk_id": vdb_result["retrieved_chunk_id"],
                "score": vdb_result["score"],
                "all_kg_chunk_ids": kg_info["all_kg_chunk_ids"],
                "edge_appearance_count": vdb_result["edge_appearance_count"],
                "chunk_appearance_count": vdb_result["chunk_appearance_count"],

            })
        else:
             logger.debug(f"Skipping VDB result for edge ('{src_id}', '{tgt_id}') as it was not found or lacked description in KG.")


    # 5. (Tùy chọn) Lấy nội dung chunk gốc từ text_chunks_db
    if text_chunks_db and final_output: # Chỉ chạy nếu có text_chunks_db và có kết quả
        all_unique_kg_chunk_ids = set()
        for item in final_output:
            all_unique_kg_chunk_ids.update(item['all_kg_chunk_ids'])

        if all_unique_kg_chunk_ids:
            logger.debug(f"Fetching content for {len(all_unique_kg_chunk_ids)} unique KG chunk IDs.")
            try:
                chunk_contents_map = {}
                # Chuyển set thành list để query
                chunk_ids_list = list(all_unique_kg_chunk_ids)
                chunk_data_list = await text_chunks_db.get_by_ids(chunk_ids_list)
                # Tạo map từ ID sang content, lọc None hoặc thiếu 'content'
                chunk_contents_map = {
                    cid: chunk_data['content']
                    for cid, chunk_data in zip(chunk_ids_list, chunk_data_list)
                    if chunk_data and 'content' in chunk_data
                }

                # Gắn content vào final_output
                # for item in final_output:
                #     item['all_kg_chunks_content'] = [
                #         chunk_contents_map.get(cid)
                #         for cid in item['all_kg_chunk_ids']
                #         if chunk_contents_map.get(cid) is not None
                #     ]
                #     # Thêm một key rỗng nếu không fetch được gì
                #     if 'all_kg_chunks_content' not in item:
                #         item['all_kg_chunks_content'] = []

            except Exception as e:
                logger.error(f"Error fetching chunk content from text_chunks_db: {e}", exc_info=True)
                # Gán giá trị mặc định hoặc None nếu lỗi
                # for item in final_output:
                #     item['all_kg_chunks_content'] = None # Hoặc [] tùy logic mong muốn
        else:
             logger.debug("No KG chunk IDs found to fetch content for.")
             # Đảm bảo key tồn tại nếu không có gì để fetch
            #  for item in final_output:
            #      item['all_kg_chunks_content'] = []


    # Sắp xếp kết quả cuối cùng theo score (distance thấp hơn là tốt hơn)
    final_output.sort(key=lambda x: x["score"])

    logger.info(f"Returning {len(final_output)} edge detail results for recall, including appearance counts.")
    return final_output

# <<< END EDGE RECALL FUNCTION >>>


# <<< START NEW DIRECT RECALL FUNCTION >>>
async def kg_direct_recall(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage, # Needed for edge recall optional content fetch
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None, # Added for keyword caching
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]: # Return candidates and keywords
    """Retrieves node and/or edge candidates directly using detailed recall functions.

    This function first extracts keywords (HL/LL) from the query, then calls
    retrieve_node_details_for_recall and/or retrieve_edge_details_for_recall
    based on the query_param.mode using the extracted keywords. It combines the results.

    Args:
        query (str): The input query string.
        knowledge_graph_inst (BaseGraphStorage): Instance of the graph storage.
        entities_vdb (BaseVectorStorage): Instance of the entity vector database.
        relationships_vdb (BaseVectorStorage): Instance of the relationship vector database.
        text_chunks_db (BaseKVStorage): Instance of the text chunk key-value store.
        query_param (QueryParam): Parameters controlling the retrieval process.
        global_config (dict[str, str]): Global configuration dictionary.
        hashing_kv (BaseKVStorage | None): Optional KV storage for keyword caching.

    Returns:
        Tuple[List[Dict[str, Any]], List[str], List[str]]: A tuple containing:
            - A list of candidate dictionaries (nodes and/or edges), each marked with 'retrieval_type'.
            - The extracted high-level keywords.
            - The extracted low-level keywords.
    """
    logger.info(f"Executing kg_direct_recall with mode: {query_param}")

    # 1. Extract Keywords
    try:
        # cache_keywords = is_query_cached(query,global_config["cache_queries"])
        if not query_param.use_query_for_retrieval:
            hl_keywords_list, ll_keywords_list = await extract_keywords_only(
                text=query,
                param=query_param,
                global_config=global_config,
                hashing_kv=hashing_kv,
                # cache_keywords=cache_keywords
            )
            hl_keywords_str = ", ".join(hl_keywords_list)
            ll_keywords_str = ", ".join(ll_keywords_list)
        else:
            ll_keywords_str, hl_keywords_str = query, query
            hl_keywords_list = [query]
            ll_keywords_list = [query]
            
        logger.debug(f"Extracted keywords for kg_direct_recall: HL='{hl_keywords_str}', LL='{ll_keywords_str}'")
    except Exception as e:
        logger.error(f"Error extracting keywords in kg_direct_recall: {e}", exc_info=True)
        return [], [], [] # Return empty lists for all parts of the tuple

    # Check if keywords are empty
    if not hl_keywords_list and not ll_keywords_list:
        logger.warning("Both HL and LL keywords are empty after extraction. Returning empty list.")
        return [], [], [] # Return empty lists for all parts of the tuple

    # Determine query strings for recall functions based on extracted keywords
    if not query_param.ll_keyword_only:
        node_query_str = ll_keywords_str or hl_keywords_str # Fallback for node query
        edge_keywords_str = hl_keywords_str # Edge recall specifically uses HL keywords
    else:
        if ll_keywords_str:
            node_query_str = ll_keywords_str
        else:
            node_query_str = hl_keywords_str

        if ll_keywords_str:
            edge_keywords_str = ll_keywords_str
        else:
            edge_keywords_str = hl_keywords_str


    # 2. Schedule Recall Tasks based on mode and available keywords
    recall_tasks = []
    node_task_idx = -1
    edge_task_idx = -1

    # Schedule Node Recall Task
    if query_param.mode in ["local", "hybrid"]:
        if node_query_str:
            recall_tasks.append(retrieve_node_details_for_recall(
                query=node_query_str,
                entities_vdb=entities_vdb,
                knowledge_graph_inst=knowledge_graph_inst,
                query_param=query_param,
            ))
            node_task_idx = len(recall_tasks) - 1
            logger.debug(f"Added node recall task for kg_direct_recall with query: '{node_query_str}'")
        else:
            logger.warning("Skipping node recall task: Effective query string is empty.")

    # Schedule Edge Recall Task
    if query_param.mode in ["global", "hybrid"]:
        if edge_keywords_str:
            recall_tasks.append(retrieve_edge_details_for_recall(
                keywords=edge_keywords_str,
                relationships_vdb=relationships_vdb,
                knowledge_graph_inst=knowledge_graph_inst,
                text_chunks_db=text_chunks_db,
                query_param=query_param
            ))
            edge_task_idx = len(recall_tasks) - 1
            logger.debug(f"Added edge recall task for kg_direct_recall with keywords: '{edge_keywords_str}'")
        else:
            logger.warning("Skipping edge recall task: HL keywords string is empty.")

    # 3. Execute Tasks
    if not recall_tasks:
        logger.warning(f"No recall tasks scheduled for mode '{query_param.mode}' based on extracted keywords. Returning empty list.")
        return [], hl_keywords_list, ll_keywords_list # Return empty candidates but keep keywords

    try:
        all_recall_results = await asyncio.gather(*recall_tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Critical error during asyncio.gather in kg_direct_recall: {e}", exc_info=True)
        return [], hl_keywords_list, ll_keywords_list # Return empty candidates but keep keywords

    # 4. Process and Combine Results (same logic as before)
    final_candidates = []

    if node_task_idx != -1:
        result = all_recall_results[node_task_idx]
        if isinstance(result, Exception):
            logger.error(f"Node recall task (kg_direct_recall) failed: {result}", exc_info=result)
        elif isinstance(result, list):
            logger.info(f"kg_direct_recall retrieved {len(result)} node candidates.")
            for node in result:
                node['retrieval_type'] = 'node'
            final_candidates.extend(result)
        else:
             logger.warning(f"Node recall task returned unexpected type: {type(result)}")

    if edge_task_idx != -1:
        result = all_recall_results[edge_task_idx]
        if isinstance(result, Exception):
            logger.error(f"Edge recall task (kg_direct_recall) failed: {result}", exc_info=result)
        elif isinstance(result, list):
            logger.info(f"kg_direct_recall retrieved {len(result)} edge candidates.")
            for edge in result:
                edge['retrieval_type'] = 'edge'
            final_candidates.extend(result)
        else:
            logger.warning(f"Edge recall task returned unexpected type: {type(result)}")

    # Sort combined list by score
    final_candidates.sort(key=lambda x: x.get('score', float('inf')), reverse=True)

    logger.info(f"kg_direct_recall completed, returning {len(final_candidates)} combined candidates after keyword extraction.")
    return final_candidates, hl_keywords_list, ll_keywords_list
# <<< END NEW DIRECT RECALL FUNCTION >>>

async def _most_relevant_text_chunks_from_nodes(
        query : str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        node: str,
        threshold: int = None
):  
    try: 
        node_data = await knowledge_graph_inst.get_node_data(node)
    except:
        return None
    list_hash_id = [compute_mdhash_id(node + des,prefix="ent-") for des in list(node_data["description"].split("<SEP>"))]
    filter_lambda = lambda x : x["__id__"] in list_hash_id

    result = await entities_vdb.query(query, top_k=1, filter_lambda = filter_lambda)
    result = result[0]
    if  threshold:
        if result["distance"] < threshold:
            return None

    return result["chunk_id"]

async def _most_relevant_text_chunks_from_edges(
        query : str,
        knowledge_graph_inst: BaseGraphStorage,
        relation_vdb: BaseVectorStorage,
        head : str, 
        tgt : str,
        threshold: int = None
):  
    try:
        edge_data = await knowledge_graph_inst.get_edge_data(head, tgt)
    except:
        try:
            edge_data = await knowledge_graph_inst.get_edge_data(tgt, head)
        except:
            return None

    list_hash_id = [compute_mdhash_id(head + tgt + des,prefix="rel-") for des in list(edge_data["description"].split("<SEP>"))]
    list_hash_id_reverse = [compute_mdhash_id(tgt + head + des,prefix="rel-") for des in list(edge_data["description"].split("<SEP>"))]

    filter_lambda = lambda x : x["__id__"] in list_hash_id
    filter_lambda_reverse = lambda x : x["__id__"] in list_hash_id_reverse


    try:
        result = await relation_vdb.query(query, top_k=1, filter_lambda = filter_lambda)
    except:
        result = await relation_vdb.query(query, top_k=1, filter_lambda = filter_lambda_reverse)

    result = result[0]
    if threshold:
        if result["distance"] < threshold:
            return None

    return  result["chunk_id"]
######### greatest result
# async def enhanced_chunk_retrieval_direct(
#     query: str,
#     knowledge_graph_inst: BaseGraphStorage,
#     entities_vdb: BaseVectorStorage,
#     relationships_vdb: BaseVectorStorage,
#     text_chunks_db: BaseKVStorage,
#     chunks_vdb: BaseVectorStorage,
#     embedding_func: callable,
#     chunk_dict,
#     doc_dict,
#     a: float = 0.3,  # Weight for chunk-query similarity
#     b: float = 0.5,  # Weight for entity score
#     c: float = 0.5,  # Weight for edge score
#     top_k_entities: int = 200,  # Number of top entities to consider
#     top_k_edges: int = 200,    # Number of top edges to consider
#     top_k_chunks: int = 100,  # Number of top chunks to return
#     use_precomputed_data: bool = True,
#     relevance_threshold = 0.7,
#     method = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Enhanced retrieval that directly loads all entities, edges, and chunks.
    
#     Process:
#     1. Calculate similarity between query and all entities
#     2. Calculate similarity between query and all edges
#     3. Select top entities and edges by similarity
#     4. Get all chunks containing these top entities and edges
#     5. For each chunk, calculate combined score using:
#        - Similarity between chunk and query (weighted by a)
#        - Sum of entity scores in that chunk (weighted by b)
#        - Sum of edge scores in that chunk (weighted by c)
#     6. Return top chunks by combined score
    
#     Args:
#         query: User's query string
#         knowledge_graph_inst: Graph storage instance
#         entities_vdb: Vector database for entities
#         relationships_vdb: Vector database for relationships
#         text_chunks_db: KV storage for text chunks
#         chunks_vdb: Vector database for chunks
#         embedding_func: Function to compute embeddings
#         chunk_dict: Dictionary of chunk information
#         doc_dict: Dictionary of document information
#         a: Weight for chunk-query similarity
#         b: Weight for entity score
#         c: Weight for edge score
#         top_k_entities: Number of top entities to consider
#         top_k_edges: Number of top edges to consider
#         top_k_chunks: Number of chunks to return
#         use_precomputed_data: Whether to use precomputed embeddings
        
#     Returns:
#         List of dictionaries with retrieved chunks and their scores
#     """
#     logger.info(f"Starting direct enhanced chunk retrieval for query: '{query}'")
    
#     # Step 1: Get query embedding
#     query_embedding = await embedding_func([query])
#     query_embedding = query_embedding[0]
    
#     # Step 2: Get all entity embeddings and compute similarity
#     entity_similarities = {}
    
#     if use_precomputed_data and hasattr(entities_vdb, "client_storage") and entities_vdb.client_storage:
#         storage_data = entities_vdb.client_storage
        
#         if "data" in storage_data and "matrix" in storage_data:
#             try:
#                 # Handle matrix format
#                 if isinstance(storage_data["matrix"], str):
#                     import base64
#                     matrix_data = np.frombuffer(base64.b64decode(storage_data["matrix"]), dtype=np.float32)
#                     embedding_dim = storage_data.get("embedding_dim", 1024)
#                     matrix = matrix_data.reshape(-1, embedding_dim)
#                 else:
#                     matrix = storage_data["matrix"]
                
#                 # Define similarity function
#                 def calculate_similarity(vec1, vec2):
#                     """Calculate cosine similarity between two vectors."""
#                     vec1 = np.squeeze(vec1)
#                     vec2 = np.squeeze(vec2)
#                     if vec1.ndim == 0 or vec2.ndim == 0 or vec1.shape[0] != vec2.shape[0]:
#                         return 0.0
#                     dot_product = np.dot(vec1, vec2)
#                     norm_vec1 = np.linalg.norm(vec1)
#                     norm_vec2 = np.linalg.norm(vec2)
#                     if norm_vec1 == 0 or norm_vec2 == 0:
#                         return 0.0
#                     return dot_product / (norm_vec1 * norm_vec2)
                
#                 query_norm = np.linalg.norm(query_embedding)
#                 if query_norm > 0:
#                     normalized_query = query_embedding / query_norm
#                 else:
#                     normalized_query = query_embedding
                    
#                 # Chuẩn hóa ma trận entity embeddings (nếu chưa được chuẩn hóa)
#                 # Tính L2 norm cho mỗi dòng (mỗi embedding)
#                 norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#                 # Tránh chia cho 0
#                 norms[norms == 0] = 1.0
#                 normalized_matrix = matrix / norms
                
#                 # Tính similarity bằng cách nhân ma trận với query vector
#                 # Đây chính là cosine similarity vì các vector đã được chuẩn hóa
#                 all_similarities = np.dot(normalized_matrix, normalized_query)
                
#                 # Map các similarities vào entity IDs và lưu thêm entity_name và description
#                 entity_info = {}  # Dictionary to store entity information
#                 for i, entity_data in enumerate(storage_data["data"]):
#                     if i < len(all_similarities) and "__id__" in entity_data:
#                         entity_id = entity_data["__id__"]
#                         entity_name = entity_data["entity_name"]
#                         entity_description = entity_data["description"]
#                         entity_similarities[entity_id] = all_similarities[i]
#                         # Store entity name and description for later use
#                         entity_info[entity_id] = {"name": entity_name, "description": entity_description}
                        
#                         # Debug: show how entity ID is calculated from name and description
#                         logger.debug(f"Entity ID '{entity_id}' is for entity named '{entity_name}'")
                        
#                 logger.info(f"Calculated similarity for {len(entity_similarities)} entities using matrix multiplication")

#             except Exception as e:
#                 logger.error(f"Error in matrix multiplication for similarities: {e}")
#                 # Fallback to original approach
#                 entity_info = {}  # Dictionary to store entity information
#                 for i, entity_data in enumerate(storage_data["data"]):
#                     if i < len(matrix) and "__id__" in entity_data:
#                         entity_id = entity_data["__id__"]
#                         entity_name = entity_data["entity_name"]
#                         entity_description = entity_data["description"]
#                         entity_embedding = matrix[i]
#                         similarity = calculate_similarity(query_embedding, entity_embedding)
#                         entity_similarities[entity_id] = similarity
#                         # Store entity name and description for later use
#                         entity_info[entity_id] = {"name": entity_name, "description": entity_description}
                        
#                         # Debug: show how entity ID is calculated from name and description
#                         logger.debug(f"Entity ID '{entity_id}' is for entity named '{entity_name}'")
    
#     if not entity_similarities:
#         logger.warning("No entity similarities calculated. Falling back to direct VDB query.")
#         entity_info = {}  # Dictionary to store entity information
#         try:
#             # Query entities VDB directly
#             entities_results = await entities_vdb.query(query, top_k=top_k_entities)
            
#             for result in entities_results:
#                 entity_id = result.get("__id__")
#                 if entity_id:
#                     # Store entity name and description
#                     entity_name = result.get("entity_name", "")
#                     entity_description = result.get("description", "")
#                     entity_info[entity_id] = {"name": entity_name, "description": entity_description}
                    
#                     # Debug: show how entity ID is calculated from name and description
#                     logger.debug(f"Entity ID '{entity_id}' is for entity named '{entity_name}'")
                    
#                     # For distance metrics (lower is better), convert to similarity
#                     if "distance" in result:
#                         entity_similarities[entity_id] = float(result["distance"])
#                     elif "score" in result:
#                         entity_similarities[entity_id] = float(result["score"])
            
#             logger.info(f"Retrieved {len(entity_similarities)} entities using VDB query")
#         except Exception as e:
#             logger.error(f"Error querying entities VDB: {e}")
#             return []
    
#     # Step 3: Sort entities by similarity and get top-k
#     sorted_entities = sorted(entity_similarities.items(), key=lambda x: x[1], reverse=True)
#     top_entities = sorted_entities[:top_k_entities]

#     if not top_entities:
#         logger.warning("No top entities found. Returning empty result.")
#         return []
    
#     logger.info(f"Selected top {len(top_entities)} entities")
    
#     # Step 4: Get edge similarities (relationships)
#     edge_similarities = {}
#     edge_info = {}
    
#     # Use similar approach to get edge similarities
#     if use_precomputed_data and hasattr(relationships_vdb, "client_storage") and relationships_vdb.client_storage:
#         storage_data = relationships_vdb.client_storage
        
#         if "data" in storage_data and "matrix" in storage_data:
#             try:
#                 # Handle matrix format
#                 if isinstance(storage_data["matrix"], str):
#                     import base64
#                     matrix_data = np.frombuffer(base64.b64decode(storage_data["matrix"]), dtype=np.float32)
#                     embedding_dim = storage_data.get("embedding_dim", 1024)
#                     matrix = matrix_data.reshape(-1, embedding_dim)
#                 else:
#                     matrix = storage_data["matrix"]
                
#                 # Calculate similarity for all edges using matrix multiplication
#                 query_norm = np.linalg.norm(query_embedding)
#                 if query_norm > 0:
#                     normalized_query = query_embedding / query_norm
#                 else:
#                     normalized_query = query_embedding
                
#                 norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#                 norms[norms == 0] = 1.0
#                 normalized_matrix = matrix / norms
                
#                 all_similarities = np.dot(normalized_matrix, normalized_query)
                
#                 for i, edge_data in enumerate(storage_data["data"]):
#                     if i < len(all_similarities) and "__id__" in edge_data:
#                         edge_id = edge_data["__id__"]
#                         src_id = edge_data.get("src_id", "")
#                         tgt_id = edge_data.get("tgt_id", "")
#                         edge_description = edge_data.get("description", "")
#                         edge_similarities[edge_id] = all_similarities[i]
#                         edge_info[edge_id] = {
#                             "src_id": src_id,
#                             "tgt_id": tgt_id,
#                             "description": edge_description
#                         }
                
#                 logger.info(f"Calculated similarity for {len(edge_similarities)} edges using matrix multiplication")
#             except Exception as e:
#                 logger.error(f"Error calculating edge similarities: {e}")
    
#     # If no edges found from precomputed data, query the VDB directly
#     if not edge_similarities:
#         logger.warning("No edge similarities calculated from precomputed data. Querying VDB directly.")
#         try:
#             # Query relationships VDB directly
#             relationships_results = await relationships_vdb.query(query, top_k=top_k_edges)
            
#             for result in relationships_results:
#                 edge_id = result.get("__id__")
#                 if edge_id:
#                     # Store edge information
#                     src_id = result.get("src_id", "")
#                     tgt_id = result.get("tgt_id", "")
#                     edge_description = result.get("description", "")
#                     edge_info[edge_id] = {
#                         "src_id": src_id,
#                         "tgt_id": tgt_id,
#                         "description": edge_description
#                     }
                    
#                     # For distance metrics (lower is better), convert to similarity
#                     if "distance" in result:
#                         edge_similarities[edge_id] = float(result["distance"])
#                     elif "score" in result:
#                         edge_similarities[edge_id] = float(result["score"])
            
#             logger.info(f"Retrieved {len(edge_similarities)} edges using VDB query")
#         except Exception as e:
#             logger.error(f"Error querying relationships VDB: {e}")
    
#     # Sort edges by similarity and get top-k
#     sorted_edges = sorted(edge_similarities.items(), key=lambda x: x[1], reverse=True)
#     top_edges = sorted_edges[:top_k_edges]
#     logger.info(f"Selected top {len(top_edges)} edges")
    
#     # Step 5: Get chunks containing these top entities
#     top_entity_ids = [entity_id for entity_id, _ in top_entities]
#     chunks_by_entity = {}
    
#     for entity_id in top_entity_ids:
#         try:
#             # Find all chunks containing this entity
#             entity_name = entity_info[entity_id]["name"]
#             node_data = await knowledge_graph_inst.get_node_data(entity_name)
#             entity_chunks = [chunk for chunk in list(node_data["source_id"].split("<SEP>")) if chunk]
#             chunks_by_entity[entity_id] = entity_chunks
#             logger.debug(f"Entity {entity_id} is found in {len(entity_chunks)} chunks")
#         except Exception as e:
#             logger.error(f"Error getting chunks for entity {entity_id}: {e}")
#             chunks_by_entity[entity_id] = []
    
#     # Step 6: Get chunks containing these top edges
#     top_edge_ids = [edge_id for edge_id, _ in top_edges]
#     chunks_by_edge = {}
    
#     for edge_id in top_edge_ids:
#         try:
#             if edge_id in edge_info:
#                 src_id = edge_info[edge_id]["src_id"]
#                 tgt_id = edge_info[edge_id]["tgt_id"]
                
#                 # Get edge data to find chunks
#                 try:
#                     edge_data = await knowledge_graph_inst.get_edge_data(src_id, tgt_id)
#                     if "source_id" in edge_data:
#                         edge_chunks = [chunk for chunk in list(edge_data["source_id"].split("<SEP>")) if chunk]
#                         chunks_by_edge[edge_id] = edge_chunks
#                         logger.debug(f"Edge {edge_id} is found in {len(edge_chunks)} chunks")
#                     else:
#                         logger.warning(f"No source_id found for edge {edge_id}")
#                         chunks_by_edge[edge_id] = []
#                 except Exception as e:
#                     logger.error(f"Error getting edge data for {src_id}->{tgt_id}: {e}")
#                     # Try reverse direction
#                     try:
#                         edge_data = await knowledge_graph_inst.get_edge_data(tgt_id, src_id)
#                         if "source_id" in edge_data:
#                             edge_chunks = [chunk for chunk in list(edge_data["source_id"].split("<SEP>")) if chunk]
#                             chunks_by_edge[edge_id] = edge_chunks
#                             logger.debug(f"Edge {edge_id} (reverse) is found in {len(edge_chunks)} chunks")
#                         else:
#                             logger.warning(f"No source_id found for edge {edge_id} (reverse)")
#                             chunks_by_edge[edge_id] = []
#                     except Exception as e2:
#                         logger.error(f"Error getting edge data for reverse direction {tgt_id}->{src_id}: {e2}")
#                         chunks_by_edge[edge_id] = []
#         except Exception as e:
#             logger.error(f"Error getting chunks for edge {edge_id}: {e}")
#             chunks_by_edge[edge_id] = []
    
#     # Step 7: Gather all unique chunks from both entities and edges
#     all_unique_chunk_ids = set()
    
#     # Add chunks from entities
#     for entity_id, chunks in chunks_by_entity.items():
#         all_unique_chunk_ids.update(chunks)
    
#     # Add chunks from edges
#     for edge_id, chunks in chunks_by_edge.items():
#         all_unique_chunk_ids.update(chunks)
    
#     if not all_unique_chunk_ids:
#         logger.warning("No chunks found containing top entities or edges. Returning empty result.")
#         return []
    
#     logger.info(f"Found {len(all_unique_chunk_ids)} unique chunks from entities and edges")
#     chunk_ids_list = list(all_unique_chunk_ids)
    
#     # Step 8: Get chunk contents
#     chunk_contents = {}
#     try:
#         chunk_data_list = await text_chunks_db.get_by_ids(chunk_ids_list)
#         for chunk_id, chunk_data in zip(chunk_ids_list, chunk_data_list):
#             if chunk_data and 'content' in chunk_data:
#                 chunk_contents[chunk_id] = chunk_data['content']
#     except Exception as e:
#         logger.error(f"Error fetching chunk contents: {e}")
#         return []
    
#     logger.info(f"Retrieved contents for {len(chunk_contents)} chunks")
    
#     # Step 9: Calculate chunk similarity to query
#     chunk_sim_scores = {}
    
#     # Try using chunks VDB first
#     try:
#         filter_lambda = lambda x: x["__id__"] in chunk_ids_list
#         vdb_results = await chunks_vdb.query(query, top_k=len(chunk_ids_list), filter_lambda=filter_lambda)
        
#         for result in vdb_results:
#             chunk_id = result.get("__id__")
#             if chunk_id:
#                 if "distance" in result:
#                     chunk_sim_scores[chunk_id] = float(result["distance"])
#                 elif "score" in result:
#                     chunk_sim_scores[chunk_id] = float(result["score"])
        
#         logger.info(f"Retrieved similarity for {len(chunk_sim_scores)} chunks using VDB")
#     except Exception as e:
#         logger.error(f"Error querying chunks VDB: {e}")
    
#     # Calculate missing similarities
#     missing_chunks = [chunk_id for chunk_id in chunk_ids_list if chunk_id not in chunk_sim_scores and chunk_id in chunk_contents]
    
#     if missing_chunks:
#         logger.info(f"Calculating similarity for {len(missing_chunks)} missing chunks")
#         try:
#             # Get embeddings for missing chunks
#             missing_contents = [chunk_contents[chunk_id] for chunk_id in missing_chunks]
#             missing_embeddings = await embedding_func(missing_contents)
            
#             for i, chunk_id in enumerate(missing_chunks):
#                 similarity = calculate_similarity(query_embedding, missing_embeddings[i])
#                 chunk_sim_scores[chunk_id] = similarity
#         except Exception as e:
#             logger.error(f"Error calculating missing chunk similarities: {e}")
    
#     # Step 10: Create mappings:
#     # - chunk_id -> list of entities in that chunk
#     # - chunk_id -> list of edges in that chunk
#     chunk_to_entities = defaultdict(list)
#     chunk_to_edges = defaultdict(list)
    
#     # Map entities to chunks
#     for entity_id, chunks in chunks_by_entity.items():
#         for chunk_id in chunks:
#             if chunk_id in all_unique_chunk_ids:
#                 chunk_to_entities[chunk_id].append(entity_id)
    
#     # Map edges to chunks
#     for edge_id, chunks in chunks_by_edge.items():
#         for chunk_id in chunks:
#             if chunk_id in all_unique_chunk_ids:
#                 chunk_to_edges[chunk_id].append(edge_id)

#     # ===== BẮT ĐẦU THÊM CODE CHO ĐỀ XUẤT 2 =====
#     # Step 10.5: Calculate relevant counts for each chunk
#     logger.info(f"Calculating relevant item counts using threshold: {relevance_threshold}")
#     chunk_relevant_counts = defaultdict(int) # Dùng defaultdict cho tiện

#     # Đếm relevant entities
#     for chunk_id, entity_ids in chunk_to_entities.items():
#         count = sum(1 for entity_id in entity_ids
#                     if entity_similarities.get(entity_id, 0.0) > relevance_threshold)
#         chunk_relevant_counts[chunk_id] += count

#     # Đếm relevant edges
#     for chunk_id, edge_ids in chunk_to_edges.items():
#         count = sum(1 for edge_id in edge_ids
#                     if edge_similarities.get(edge_id, 0.0) > relevance_threshold)
#         chunk_relevant_counts[chunk_id] += count

#     logger.info(f"Calculated relevant counts for {len(chunk_relevant_counts)} chunks")
#     # ===== KẾT THÚC THÊM CODE CHO ĐỀ XUẤT 2 =====

#     # Step 11: Calculate entity and edge scores for each chunk
#     # Instead of averages, we use the sum of scores
#     chunk_entity_scores = {}
#     chunk_edge_scores = {}
    
#     # Calculate sum of entity scores for each chunk
#     for chunk_id, entity_ids in chunk_to_entities.items():
#         entity_scores = [entity_similarities[entity_id] for entity_id in entity_ids if entity_id in entity_similarities]
#         if entity_scores:
#             chunk_entity_scores[chunk_id] = sum(entity_scores) / len(entity_scores)
#             # chunk_entity_scores[chunk_id] = sum(entity_scores) 
#         else:
#             chunk_entity_scores[chunk_id] = 0.0
    
#     # Calculate sum of edge scores for each chunk
#     for chunk_id, edge_ids in chunk_to_edges.items():
#         edge_scores = [edge_similarities[edge_id] for edge_id in edge_ids if edge_id in edge_similarities]
#         if edge_scores:
#             # Use sum instead of average
#             # chunk_edge_scores[chunk_id] = sum(edge_scores)
#             chunk_edge_scores[chunk_id] = sum(edge_scores) / len(edge_scores)
#         else:
#             chunk_edge_scores[chunk_id] = 0.0
    
#     # Step 12: Normalize scores
#     def normalize_scores(scores):
#         if not scores:
#             return {}
#         values = list(scores.values())
#         min_val = min(values)
#         max_val = max(values)
#         range_val = max_val - min_val
#         if range_val == 0:
#             return {k: 1.0 for k in scores}
#         return {k: (v - min_val) / range_val for k, v in scores.items()}
    
#     normalized_chunk_sim = normalize_scores(chunk_sim_scores)
#     normalized_entity_scores = normalize_scores(chunk_entity_scores)
#     normalized_edge_scores = normalize_scores(chunk_edge_scores)
    
#     import math
#     # Step 13: Combine scores from all three sources
#     final_scores = {}
#     for chunk_id in all_unique_chunk_ids:
#         # Get component scores, using 0.0 for missing scores
#         chunk_score = normalized_chunk_sim.get(chunk_id, 0.0)
#         entity_score = normalized_entity_scores.get(chunk_id, 0.0)
#         edge_score = normalized_edge_scores.get(chunk_id, 0.0)
        
#         # Combine using weighted sum (a + b + c should equal 1.0)
#         # final_scores[chunk_id] = (
#         #     a * chunk_score + 
#         #     b * entity_score + 
#         #     c * edge_score
#         # )

# # #################### New Method ######################
#         base_score = (
#             a * chunk_score +
#             b * entity_score +
#             c * edge_score
#         )
    
#         # ===== BẮT ĐẦU SỬA ĐỔI STEP 13 =====
#         # Get the relevant count for this chunk
#         relevant_count = chunk_relevant_counts.get(chunk_id, 0)

#         # Calculate the modifier using a non-linear function (log1p)
#         # math.log1p(x) tính ln(1 + x), đảm bảo kết quả >= 0 và xử lý x=0
#         modifier = math.log1p(relevant_count)

#         # Apply the modifier to the base score
#         # Lựa chọn 1: Nhân (khuếch đại điểm) - Giống gợi ý trong tài liệu hơn
#         # final_scores[chunk_id] = base_score * (1 + modifier)
        
#         density_bonus_weight = 0.1 # Có thể thêm một trọng số cho bonus
#         final_scores[chunk_id] = base_score + density_bonus_weight * modifier
#     # Step 14: Get final result chunks
#     result_chunks = []
#     sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_chunks]
    
#     for chunk_id, score in sorted_chunks:
#         if chunk_id in chunk_contents:
#             doc_id = chunk_dict[chunk_id]['full_doc_id']
#             doc_content = doc_dict[doc_id]['content']
#             result_chunks.append({
#                 "id": chunk_id,
#                 "chunk_content": chunk_contents[chunk_id],
#                 "doc_id": doc_id,
#                 "doc_content": doc_content,
#                 "score": score,
#                 "chunk_sim": normalized_chunk_sim.get(chunk_id, 0.0),
#                 "entity_score": normalized_entity_scores.get(chunk_id, 0.0),
#                 "edge_score": normalized_edge_scores.get(chunk_id, 0.0),
#                 "entities": chunk_to_entities.get(chunk_id, []),
#                 "edges": chunk_to_edges.get(chunk_id, [])
#             })
    
#     logger.info(f"Enhanced direct chunk retrieval completed, returning {len(result_chunks)} chunks")
#     return result_chunks


async def enhanced_chunk_retrieval_direct(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    chunks_vdb: BaseVectorStorage,
    embedding_func: callable,
    chunk_dict,
    doc_dict,
    a: float = 0.3,  # Weight for chunk-query similarity
    b: float = 0.5,  # Weight for entity score
    c: float = 0.5,  # Weight for edge score
    top_k_entities: int = 200,  # Number of top entities to consider
    top_k_edges: int = 200,    # Number of top edges to consider
    top_k_chunks: int = 100,  # Number of top chunks to return
    use_precomputed_data: bool = True,
    relevance_threshold = 0.7,
    method = None,
) -> List[Dict[str, Any]]:
    """
    Enhanced retrieval that directly loads all entities, edges, and chunks.
    
    Process:
    1. Calculate similarity between query and all entities
    2. Calculate similarity between query and all edges
    3. Select top entities and edges by similarity
    4. Get all chunks containing these top entities and edges
    5. For each chunk, calculate combined score using:
       - Similarity between chunk and query (weighted by a)
       - Sum of entity scores in that chunk (weighted by b)
       - Sum of edge scores in that chunk (weighted by c)
    6. Return top chunks by combined score
    
    Args:
        query: User's query string
        knowledge_graph_inst: Graph storage instance
        entities_vdb: Vector database for entities
        relationships_vdb: Vector database for relationships
        text_chunks_db: KV storage for text chunks
        chunks_vdb: Vector database for chunks
        embedding_func: Function to compute embeddings
        chunk_dict: Dictionary of chunk information
        doc_dict: Dictionary of document information
        a: Weight for chunk-query similarity
        b: Weight for entity score
        c: Weight for edge score
        top_k_entities: Number of top entities to consider
        top_k_edges: Number of top edges to consider
        top_k_chunks: Number of chunks to return
        use_precomputed_data: Whether to use precomputed embeddings
        
    Returns:
        List of dictionaries with retrieved chunks and their scores
    """
    logger.info(f"Starting direct enhanced chunk retrieval for query: '{query}'")
    
    # Step 1: Get query embedding
    query_embedding = await embedding_func([query])
    query_embedding = query_embedding[0]
    
    # Step 2: Get all entity embeddings and compute similarity
    entity_similarities = {}
    
    if use_precomputed_data and hasattr(entities_vdb, "client_storage") and entities_vdb.client_storage:
        storage_data = entities_vdb.client_storage
        
        if "data" in storage_data and "matrix" in storage_data:
            try:
                # Handle matrix format
                if isinstance(storage_data["matrix"], str):
                    import base64
                    matrix_data = np.frombuffer(base64.b64decode(storage_data["matrix"]), dtype=np.float32)
                    embedding_dim = storage_data.get("embedding_dim", 1024)
                    matrix = matrix_data.reshape(-1, embedding_dim)
                else:
                    matrix = storage_data["matrix"]
                
                # Define similarity function
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
                
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0:
                    normalized_query = query_embedding / query_norm
                else:
                    normalized_query = query_embedding
                    
                # Chuẩn hóa ma trận entity embeddings (nếu chưa được chuẩn hóa)
                # Tính L2 norm cho mỗi dòng (mỗi embedding)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                # Tránh chia cho 0
                norms[norms == 0] = 1.0
                normalized_matrix = matrix / norms
                
                # Tính similarity bằng cách nhân ma trận với query vector
                # Đây chính là cosine similarity vì các vector đã được chuẩn hóa
                all_similarities = np.dot(normalized_matrix, normalized_query)
                
                # Map các similarities vào entity IDs và lưu thêm entity_name và description
                entity_info = {}  # Dictionary to store entity information
                for i, entity_data in enumerate(storage_data["data"]):
                    if i < len(all_similarities) and "__id__" in entity_data:
                        entity_id = entity_data["__id__"]
                        entity_name = entity_data["entity_name"]
                        entity_description = entity_data["description"]
                        entity_similarities[entity_id] = all_similarities[i]
                        # Store entity name and description for later use
                        entity_info[entity_id] = {"name": entity_name, "description": entity_description}
                        
                        # Debug: show how entity ID is calculated from name and description
                        logger.debug(f"Entity ID '{entity_id}' is for entity named '{entity_name}'")
                        
                logger.info(f"Calculated similarity for {len(entity_similarities)} entities using matrix multiplication")

            except Exception as e:
                logger.error(f"Error in matrix multiplication for similarities: {e}")
                # Fallback to original approach
                entity_info = {}  # Dictionary to store entity information
                for i, entity_data in enumerate(storage_data["data"]):
                    if i < len(matrix) and "__id__" in entity_data:
                        entity_id = entity_data["__id__"]
                        entity_name = entity_data["entity_name"]
                        entity_description = entity_data["description"]
                        entity_embedding = matrix[i]
                        similarity = calculate_similarity(query_embedding, entity_embedding)
                        entity_similarities[entity_id] = similarity
                        # Store entity name and description for later use
                        entity_info[entity_id] = {"name": entity_name, "description": entity_description}
                        
                        # Debug: show how entity ID is calculated from name and description
                        logger.debug(f"Entity ID '{entity_id}' is for entity named '{entity_name}'")
    
    if not entity_similarities:
        logger.warning("No entity similarities calculated. Falling back to direct VDB query.")
        entity_info = {}  # Dictionary to store entity information
        try:
            # Query entities VDB directly
            entities_results = await entities_vdb.query(query, top_k=top_k_entities)
            
            for result in entities_results:
                entity_id = result.get("__id__")
                if entity_id:
                    # Store entity name and description
                    entity_name = result.get("entity_name", "")
                    entity_description = result.get("description", "")
                    entity_info[entity_id] = {"name": entity_name, "description": entity_description}
                    
                    # Debug: show how entity ID is calculated from name and description
                    logger.debug(f"Entity ID '{entity_id}' is for entity named '{entity_name}'")
                    
                    # For distance metrics (lower is better), convert to similarity
                    if "distance" in result:
                        entity_similarities[entity_id] = float(result["distance"])
                    elif "score" in result:
                        entity_similarities[entity_id] = float(result["score"])
            
            logger.info(f"Retrieved {len(entity_similarities)} entities using VDB query")
        except Exception as e:
            logger.error(f"Error querying entities VDB: {e}")
            return []
    
    # Step 3: Sort entities by similarity and get top-k
    sorted_entities = sorted(entity_similarities.items(), key=lambda x: x[1], reverse=True)
    top_entities = sorted_entities[:top_k_entities]

    if not top_entities:
        logger.warning("No top entities found. Returning empty result.")
        return []
    
    logger.info(f"Selected top {len(top_entities)} entities")
    
    # Step 4: Get edge similarities (relationships)
    edge_similarities = {}
    edge_info = {}
    
    # Use similar approach to get edge similarities
    if use_precomputed_data and hasattr(relationships_vdb, "client_storage") and relationships_vdb.client_storage:
        storage_data = relationships_vdb.client_storage
        
        if "data" in storage_data and "matrix" in storage_data:
            try:
                # Handle matrix format
                if isinstance(storage_data["matrix"], str):
                    import base64
                    matrix_data = np.frombuffer(base64.b64decode(storage_data["matrix"]), dtype=np.float32)
                    embedding_dim = storage_data.get("embedding_dim", 1024)
                    matrix = matrix_data.reshape(-1, embedding_dim)
                else:
                    matrix = storage_data["matrix"]
                
                # Calculate similarity for all edges using matrix multiplication
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0:
                    normalized_query = query_embedding / query_norm
                else:
                    normalized_query = query_embedding
                
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normalized_matrix = matrix / norms
                
                all_similarities = np.dot(normalized_matrix, normalized_query)
                
                for i, edge_data in enumerate(storage_data["data"]):
                    if i < len(all_similarities) and "__id__" in edge_data:
                        edge_id = edge_data["__id__"]
                        src_id = edge_data.get("src_id", "")
                        tgt_id = edge_data.get("tgt_id", "")
                        edge_description = edge_data.get("description", "")
                        edge_similarities[edge_id] = all_similarities[i]
                        edge_info[edge_id] = {
                            "src_id": src_id,
                            "tgt_id": tgt_id,
                            "description": edge_description
                        }
                
                logger.info(f"Calculated similarity for {len(edge_similarities)} edges using matrix multiplication")
            except Exception as e:
                logger.error(f"Error calculating edge similarities: {e}")
    
    # If no edges found from precomputed data, query the VDB directly
    if not edge_similarities:
        logger.warning("No edge similarities calculated from precomputed data. Querying VDB directly.")
        try:
            # Query relationships VDB directly
            relationships_results = await relationships_vdb.query(query, top_k=top_k_edges)
            
            for result in relationships_results:
                edge_id = result.get("__id__")
                if edge_id:
                    # Store edge information
                    src_id = result.get("src_id", "")
                    tgt_id = result.get("tgt_id", "")
                    edge_description = result.get("description", "")
                    edge_info[edge_id] = {
                        "src_id": src_id,
                        "tgt_id": tgt_id,
                        "description": edge_description
                    }
                    
                    # For distance metrics (lower is better), convert to similarity
                    if "distance" in result:
                        edge_similarities[edge_id] = float(result["distance"])
                    elif "score" in result:
                        edge_similarities[edge_id] = float(result["score"])
            
            logger.info(f"Retrieved {len(edge_similarities)} edges using VDB query")
        except Exception as e:
            logger.error(f"Error querying relationships VDB: {e}")
    
    # Sort edges by similarity and get top-k
    sorted_edges = sorted(edge_similarities.items(), key=lambda x: x[1], reverse=True)
    top_edges = sorted_edges[:top_k_edges]
    logger.info(f"Selected top {len(top_edges)} edges")
    
    # Step 5: Get chunks containing these top entities
    top_entity_ids = [entity_id for entity_id, _ in top_entities]
    chunks_by_entity = {}
    
    for entity_id in top_entity_ids:
        try:
            # Find all chunks containing this entity
            entity_name = entity_info[entity_id]["name"]
            node_data = await knowledge_graph_inst.get_node_data(entity_name)
            entity_chunks = [chunk for chunk in list(node_data["source_id"].split("<SEP>")) if chunk]
            chunks_by_entity[entity_id] = entity_chunks
            logger.debug(f"Entity {entity_id} is found in {len(entity_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error getting chunks for entity {entity_id}: {e}")
            chunks_by_entity[entity_id] = []
    
    # Step 6: Get chunks containing these top edges
    top_edge_ids = [edge_id for edge_id, _ in top_edges]
    chunks_by_edge = {}
    
    for edge_id in top_edge_ids:
        try:
            if edge_id in edge_info:
                src_id = edge_info[edge_id]["src_id"]
                tgt_id = edge_info[edge_id]["tgt_id"]
                
                # Get edge data to find chunks
                try:
                    edge_data = await knowledge_graph_inst.get_edge_data(src_id, tgt_id)
                    if "source_id" in edge_data:
                        edge_chunks = [chunk for chunk in list(edge_data["source_id"].split("<SEP>")) if chunk]
                        chunks_by_edge[edge_id] = edge_chunks
                        logger.debug(f"Edge {edge_id} is found in {len(edge_chunks)} chunks")
                    else:
                        logger.warning(f"No source_id found for edge {edge_id}")
                        chunks_by_edge[edge_id] = []
                except Exception as e:
                    logger.error(f"Error getting edge data for {src_id}->{tgt_id}: {e}")
                    # Try reverse direction
                    try:
                        edge_data = await knowledge_graph_inst.get_edge_data(tgt_id, src_id)
                        if "source_id" in edge_data:
                            edge_chunks = [chunk for chunk in list(edge_data["source_id"].split("<SEP>")) if chunk]
                            chunks_by_edge[edge_id] = edge_chunks
                            logger.debug(f"Edge {edge_id} (reverse) is found in {len(edge_chunks)} chunks")
                        else:
                            logger.warning(f"No source_id found for edge {edge_id} (reverse)")
                            chunks_by_edge[edge_id] = []
                    except Exception as e2:
                        logger.error(f"Error getting edge data for reverse direction {tgt_id}->{src_id}: {e2}")
                        chunks_by_edge[edge_id] = []
        except Exception as e:
            logger.error(f"Error getting chunks for edge {edge_id}: {e}")
            chunks_by_edge[edge_id] = []
    
    # Step 7: Gather all unique chunks from both entities and edges
    all_unique_chunk_ids = set()
    
    # Add chunks from entities
    for entity_id, chunks in chunks_by_entity.items():
        all_unique_chunk_ids.update(chunks)
    
    # Add chunks from edges
    for edge_id, chunks in chunks_by_edge.items():
        all_unique_chunk_ids.update(chunks)
    
    if not all_unique_chunk_ids:
        logger.warning("No chunks found containing top entities or edges. Returning empty result.")
        return []
    
    logger.info(f"Found {len(all_unique_chunk_ids)} unique chunks from entities and edges")
    chunk_ids_list = list(all_unique_chunk_ids)
    
    # Step 8: Get chunk contents
    chunk_contents = {}
    try:
        chunk_data_list = await text_chunks_db.get_by_ids(chunk_ids_list)
        for chunk_id, chunk_data in zip(chunk_ids_list, chunk_data_list):
            if chunk_data and 'content' in chunk_data:
                chunk_contents[chunk_id] = chunk_data['content']
    except Exception as e:
        logger.error(f"Error fetching chunk contents: {e}")
        return []
    
    logger.info(f"Retrieved contents for {len(chunk_contents)} chunks")
    
    # Step 9: Calculate chunk similarity to query
    chunk_sim_scores = {}
    
    # Try using chunks VDB first
    try:
        filter_lambda = lambda x: x["__id__"] in chunk_ids_list
        vdb_results = await chunks_vdb.query(query, top_k=len(chunk_ids_list), filter_lambda=filter_lambda)
        
        for result in vdb_results:
            chunk_id = result.get("__id__")
            if chunk_id:
                if "distance" in result:
                    chunk_sim_scores[chunk_id] = float(result["distance"])
                elif "score" in result:
                    chunk_sim_scores[chunk_id] = float(result["score"])
        
        logger.info(f"Retrieved similarity for {len(chunk_sim_scores)} chunks using VDB")
    except Exception as e:
        logger.error(f"Error querying chunks VDB: {e}")
    
    # Calculate missing similarities
    missing_chunks = [chunk_id for chunk_id in chunk_ids_list if chunk_id not in chunk_sim_scores and chunk_id in chunk_contents]
    
    if missing_chunks:
        logger.info(f"Calculating similarity for {len(missing_chunks)} missing chunks")
        try:
            # Get embeddings for missing chunks
            missing_contents = [chunk_contents[chunk_id] for chunk_id in missing_chunks]
            missing_embeddings = await embedding_func(missing_contents)
            
            for i, chunk_id in enumerate(missing_chunks):
                similarity = calculate_similarity(query_embedding, missing_embeddings[i])
                chunk_sim_scores[chunk_id] = similarity
        except Exception as e:
            logger.error(f"Error calculating missing chunk similarities: {e}")
    
    # Step 10: Create mappings:
    # - chunk_id -> list of entities in that chunk
    # - chunk_id -> list of edges in that chunk
    chunk_to_entities = defaultdict(list)
    chunk_to_edges = defaultdict(list)
    
    # Map entities to chunks
    for entity_id, chunks in chunks_by_entity.items():
        for chunk_id in chunks:
            if chunk_id in all_unique_chunk_ids:
                chunk_to_entities[chunk_id].append(entity_id)
    
    # Map edges to chunks
    for edge_id, chunks in chunks_by_edge.items():
        for chunk_id in chunks:
            if chunk_id in all_unique_chunk_ids:
                chunk_to_edges[chunk_id].append(edge_id)

    # ===== BẮT ĐẦU THÊM CODE CHO ĐỀ XUẤT 2 =====
    # Step 10.5: Calculate relevant counts for each chunk
    logger.info(f"Calculating relevant item counts using threshold: {relevance_threshold}")
    chunk_relevant_counts = defaultdict(int) # Dùng defaultdict cho tiện

    # Đếm relevant entities
    for chunk_id, entity_ids in chunk_to_entities.items():
        count = sum(1 for entity_id in entity_ids
                    if entity_similarities.get(entity_id, 0.0) > relevance_threshold)
        chunk_relevant_counts[chunk_id] += count

    # Đếm relevant edges
    for chunk_id, edge_ids in chunk_to_edges.items():
        count = sum(1 for edge_id in edge_ids
                    if edge_similarities.get(edge_id, 0.0) > relevance_threshold)
        chunk_relevant_counts[chunk_id] += count

    logger.info(f"Calculated relevant counts for {len(chunk_relevant_counts)} chunks")
    # ===== KẾT THÚC THÊM CODE CHO ĐỀ XUẤT 2 =====

    # Step 11: Calculate entity and edge scores for each chunk
    # Instead of averages, we use the sum of scores
    chunk_entity_scores = {}
    chunk_edge_scores = {}
    
    # Calculate sum of entity scores for each chunk
    for chunk_id, entity_ids in chunk_to_entities.items():
        entity_scores = [entity_similarities[entity_id] for entity_id in entity_ids if entity_id in entity_similarities]
        if entity_scores:
            chunk_entity_scores[chunk_id] = sum(entity_scores) / len(entity_scores)
            # chunk_entity_scores[chunk_id] = sum(entity_scores) 
        else:
            chunk_entity_scores[chunk_id] = 0.0
    
    # Calculate sum of edge scores for each chunk
    for chunk_id, edge_ids in chunk_to_edges.items():
        edge_scores = [edge_similarities[edge_id] for edge_id in edge_ids if edge_id in edge_similarities]
        if edge_scores:
            # Use sum instead of average
            # chunk_edge_scores[chunk_id] = sum(edge_scores)
            chunk_edge_scores[chunk_id] = sum(edge_scores) / len(edge_scores)
        else:
            chunk_edge_scores[chunk_id] = 0.0
    
    # Step 12: Normalize scores
    def normalize_scores(scores):
        if not scores:
            return {}
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        if range_val == 0:
            return {k: 1.0 for k in scores}
        return {k: (v - min_val) / range_val for k, v in scores.items()}
    
    normalized_chunk_sim = normalize_scores(chunk_sim_scores)
    normalized_entity_scores = normalize_scores(chunk_entity_scores)
    normalized_edge_scores = normalize_scores(chunk_edge_scores)
    
    import math
    # Step 13: Combine scores from all three sources
    final_scores = {}
    for chunk_id in all_unique_chunk_ids:
        # Get component scores, using 0.0 for missing scores
        chunk_score = normalized_chunk_sim.get(chunk_id, 0.0)
        entity_score = normalized_entity_scores.get(chunk_id, 0.0)
        edge_score = normalized_edge_scores.get(chunk_id, 0.0)
        
        # Use different scoring methods based on the 'method' parameter
        if method == "max":
            # Get entities and edges for this chunk
            entities = chunk_to_entities.get(chunk_id, [])
            edges = chunk_to_edges.get(chunk_id, [])
            
            # Get raw entity scores above threshold
            entity_scores_above_threshold = [
                entity_similarities[entity_id]
                for entity_id in entities 
                if entity_id in entity_similarities and entity_similarities[entity_id] > relevance_threshold
            ]
            
            # Get raw edge scores above threshold
            edge_scores_above_threshold = [
                edge_similarities[edge_id]
                for edge_id in edges 
                if edge_id in edge_similarities and edge_similarities[edge_id] > relevance_threshold
            ]
            
            # Calculate max scores (if any) using raw values
            max_entity_score = max(entity_scores_above_threshold) if entity_scores_above_threshold else 0
            max_edge_score = max(edge_scores_above_threshold) if edge_scores_above_threshold else 0
            raw_chunk_score = chunk_sim_scores.get(chunk_id, 0.0)
            
            # Calculate final score using raw values
            final_scores[chunk_id] = a * max_entity_score + b * max_edge_score + c * chunk_score
            
        elif method == "max_top_k":
            # Get entities and edges for this chunk
            entities = chunk_to_entities.get(chunk_id, [])
            edges = chunk_to_edges.get(chunk_id, [])
            
            # Lọc entity dựa trên điểm raw > threshold
            entities_above_threshold = [
                entity_id for entity_id in entities 
                if entity_id in entity_similarities and entity_similarities[entity_id] > relevance_threshold
            ]
            
            # Lọc edge dựa trên điểm raw > threshold
            edges_above_threshold = [
                edge_id for edge_id in edges 
                if edge_id in edge_similarities and edge_similarities[edge_id] > relevance_threshold
            ]
            
            # Lấy raw scores của entities vượt ngưỡng và sắp xếp
            entity_raw_scores = [(entity_id, entity_similarities[entity_id]) 
                              for entity_id in entities_above_threshold]
            entity_raw_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy raw scores của edges vượt ngưỡng và sắp xếp
            edge_raw_scores = [(edge_id, edge_similarities[edge_id])
                            for edge_id in edges_above_threshold]
            edge_raw_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy top 5 (hoặc tất cả nếu ít hơn 5) và tính trung bình
            top_entity_scores = [score for _, score in entity_raw_scores[:5]]
            top_edge_scores = [score for _, score in edge_raw_scores[:5]]
            
            # Lấy điểm raw của chunk - không dùng điểm chuẩn hóa
            raw_chunk_score = chunk_sim_scores.get(chunk_id, 0.0)
            
            # Tính trung bình điểm raw cho entity và edge
            avg_entity_score = sum(top_entity_scores) / len(top_entity_scores) if top_entity_scores else 0
            avg_edge_score = sum(top_edge_scores) / len(top_edge_scores) if top_edge_scores else 0
            
            # Tính điểm cuối cùng chỉ dùng điểm raw
            final_scores[chunk_id] = a * avg_entity_score + b * avg_edge_score + c * raw_chunk_score
            
        elif method == "new_function":
            # Use the current implementation with the log1p modifier
            base_score = (
                a * chunk_score +
                b * entity_score +
                c * edge_score
            )
            
            # Get the relevant count for this chunk
            relevant_count = chunk_relevant_counts.get(chunk_id, 0)
            
            # Calculate the modifier using a non-linear function (log1p)
            modifier = math.log1p(relevant_count)
            
            # Apply the modifier to the base score
            final_scores[chunk_id] = base_score * (1 + modifier)
            
        else:  # method == None or any other value
            # Use the original weighted sum formula (from commented code)
            final_scores[chunk_id] = (
                a * chunk_score + 
                b * entity_score + 
                c * edge_score
            )

    # Step 14: Get final result chunks
    result_chunks = []
    sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_chunks]
    
    for chunk_id, score in sorted_chunks:
        if chunk_id in chunk_contents:
            doc_id = chunk_dict[chunk_id]['full_doc_id']
            doc_content = doc_dict[doc_id]['content']
            result_chunks.append({
                "id": chunk_id,
                "chunk_content": chunk_contents[chunk_id],
                "doc_id": doc_id,
                "doc_content": doc_content,
                "score": score,
                "chunk_sim": normalized_chunk_sim.get(chunk_id, 0.0),
                "entity_score": normalized_entity_scores.get(chunk_id, 0.0),
                "edge_score": normalized_edge_scores.get(chunk_id, 0.0),
                "entities": chunk_to_entities.get(chunk_id, []),
                "edges": chunk_to_edges.get(chunk_id, [])
            })
    
    logger.info(f"Enhanced direct chunk retrieval completed, returning {len(result_chunks)} chunks")
    return result_chunks