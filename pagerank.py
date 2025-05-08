import networkx as nx
import numpy as np
from collections import defaultdict
import json

# --- CÁC HÀM PLACEHOLDER BẠN CẦN IMPLEMENT VỚI NANO_VECTORDB ---
def get_query_embedding(query_text: str, embedding_type: str = "passage"):
    """Lấy embedding cho câu truy vấn.
    embedding_type có thể là 'passage' hoặc 'fact' để có thể tạo embedding khác nhau.
    """
    # TODO: Implement using nano_vectordb
    print(f"[TODO] Getting query embedding for: '{query_text}' (type: {embedding_type})")
    # Giả sử trả về một numpy array
    if embedding_type == "fact":
        return np.random.rand(1, 768) # Kích thước embedding ví dụ
    return np.random.rand(1, 768)

def get_all_fact_identifiers_and_embeddings(G_nx):
    """
    Lấy tất cả 'fact' identifiers và embeddings của chúng.
    'Fact' ở đây có thể là các cạnh (edges) trong đồ thị của bạn.
    """
    # TODO: Implement using nano_vectordb
    # Ví dụ: nếu facts là các cạnh
    facts = {}
    print(f"[TODO] Getting embeddings for all {len(G_nx.edges())} edges as facts.")
    for u, v in G_nx.edges():
        # Giả sử edge tuple (u,v) là identifier, hoặc bạn có ID riêng cho edge
        edge_id = tuple(sorted((u, v))) # Để đảm bảo tính duy nhất cho đồ thị vô hướng
        # facts[edge_id] = get_edge_embedding_from_db(u,v) # Hàm bạn tự định nghĩa
        facts[edge_id] = np.random.rand(1, 768) # Placeholder
    return facts

def get_all_passage_ids_and_embeddings(all_chunk_data_keys):
    """Lấy tất cả passage_ids (chunk_ids) và embeddings."""
    # TODO: Implement using nano_vectordb
    passage_embeddings = {}
    print(f"[TODO] Getting embeddings for all {len(all_chunk_data_keys)} passages.")
    for chunk_id in all_chunk_data_keys:
        # passage_embeddings[chunk_id] = get_passage_embedding_from_db(chunk_id)
        passage_embeddings[chunk_id] = np.random.rand(1, 768) # Placeholder
    return passage_embeddings

def calculate_similarity(vec1, vec2):
    # TODO: Implement similarity (ví dụ: cosine similarity)
    # Đảm bảo vec1 và vec2 là 1D arrays hoặc xử lý cho phù hợp
    vec1 = np.squeeze(vec1)
    vec2 = np.squeeze(vec2)
    if vec1.ndim == 0 or vec2.ndim == 0 or vec1.shape[0] != vec2.shape[0]:
        # print(f"Warning: Invalid vector shapes for similarity: {vec1.shape}, {vec2.shape}")
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def min_max_normalize_dict_scores(scores_dict):
    if not scores_dict:
        return {}
    values = np.array(list(scores_dict.values()))
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    if range_val == 0: # Tất cả các giá trị bằng nhau
        return {k: 1.0 for k in scores_dict}
    return {k: (v - min_val) / range_val for k, v in scores_dict.items()}

# --- HÀM TRUY XUẤT CHÍNH ---
def reproduce_hipporag_retrieval_nx(
    G_nx: nx.Graph,
    query: str,
    all_chunk_data: dict, # Ví dụ: {chunk_id: {'content': "...", 'entity_ids_in_chunk': [entity_A, entity_B]}}
    synonym_data: dict,   # Ví dụ: {entity_id: [synonym_id_1, ...]}
    config: dict
):
    """
    Reproduces the core HippoRAG retrieval logic using NetworkX.

    Args:
        G_nx: NetworkX graph (undirected, edges have 'weight').
              Nodes are entity IDs. Nodes may have 'chunk_ids' attribute (list).
        query: The user's query string.
        all_chunk_data: Dict mapping chunk_id to its data (content, entities it contains).
        synonym_data: Dict mapping entity_id to list of its synonym_ids.
        config: Dictionary for hyperparameters:
            - linking_top_k_facts: Number of top facts to consider.
            - passage_node_weight_factor: Factor to weigh DPR scores for personalization. (HippoRAG's passage_node_weight)
            - damping_factor: Damping factor for PageRank (e.g., 0.5 in HippoRAG).
            - num_retrieved_chunks: Number of chunks to return.
    """
    print(f"\n--- Starting Retrieval for Query: '{query}' ---")

    # 1. Get Query Embeddings
    query_embedding_for_fact = get_query_embedding(query, embedding_type="fact")
    query_embedding_for_passage = get_query_embedding(query, embedding_type="passage")
    print("1. Query embeddings generated.")

    # 2. Fact Scoring and Simplified Reranking
    # Giả sử "facts" là các cạnh của đồ thị.
    all_facts_with_embeddings = get_all_fact_identifiers_and_embeddings(G_nx)
    query_fact_scores = {}
    for fact_id, fact_emb in all_facts_with_embeddings.items():
        query_fact_scores[fact_id] = calculate_similarity(query_embedding_for_fact, fact_emb)

    # Sắp xếp facts và lấy top K (Simplified Reranking)
    sorted_facts = sorted(query_fact_scores.items(), key=lambda item: item[1], reverse=True)
    top_k_candidate_facts_with_scores = sorted_facts[:config.get('linking_top_k_facts', 10)]
    # top_k_candidate_facts bây giờ là list của (fact_id, score)
    # fact_id ở đây là tuple (u,v) của cạnh
    print(f"2. Scored {len(all_facts_with_embeddings)} facts. Top {len(top_k_candidate_facts_with_scores)} facts selected.")

    # 3. Dense Passage Retrieval (DPR)
    all_passage_ids = list(all_chunk_data.keys())
    all_passage_embeddings_map = get_all_passage_ids_and_embeddings(all_passage_ids)
    dpr_scores = {}
    for passage_id, passage_emb in all_passage_embeddings_map.items():
        dpr_scores[passage_id] = calculate_similarity(query_embedding_for_passage, passage_emb)
    normalized_dpr_scores = min_max_normalize_dict_scores(dpr_scores)
    print("3. Dense Passage Retrieval (DPR) scores calculated and normalized.")

    # 4. Prepare Personalization Vector for PageRank (targeting entity nodes in G_nx)
    personalization_vector = {node: 0.0 for node in G_nx.nodes()}

    # 4a. Add weights from Facts/Edges
    # Các entity node tham gia vào top_k_facts sẽ nhận trọng số từ điểm của fact đó
    if top_k_candidate_facts_with_scores:
        # Chuẩn hóa điểm của top_k_facts để tổng các đóng góp từ fact vào personalization là hợp lý
        fact_contribution_scores = {fact_id: score for fact_id, score in top_k_candidate_facts_with_scores}
        normalized_fact_contribution_scores = min_max_normalize_dict_scores(fact_contribution_scores)

        for fact_id, normalized_score in normalized_fact_contribution_scores.items():
            # fact_id là một tuple (u,v) đại diện cho một cạnh
            u, v = fact_id
            if u in personalization_vector:
                personalization_vector[u] += normalized_score
            if v in personalization_vector:
                personalization_vector[v] += normalized_score
    print("4a. Weights from facts added to personalization vector.")

    # 4b. Add weights from Passage DPR scores (influencing entity nodes)
    # Các entity node sẽ nhận thêm trọng số nếu chúng thuộc về các passage có điểm DPR cao.
    passage_influence_factor = config.get('passage_node_weight_factor', 0.1) # Hệ số ảnh hưởng của passage

    # Cách 1: Nếu node có thuộc tính chunk_ids
    # for node_id in G_nx.nodes():
    #     associated_chunk_ids = G_nx.nodes[node_id].get('chunk_ids', []) # Giả sử node có attr này
    #     passage_score_for_node = 0
    #     for chunk_id in associated_chunk_ids:
    #         passage_score_for_node += normalized_dpr_scores.get(chunk_id, 0.0)
    #     if associated_chunk_ids: # Trung bình điểm passage
    #        personalization_vector[node_id] += (passage_score_for_node / len(associated_chunk_ids)) * passage_influence_factor

    # Cách 2: Dùng all_chunk_data để biết entity nào thuộc passage nào
    entity_to_dpr_passage_score_sum = defaultdict(float)
    entity_passage_counts = defaultdict(int)

    for chunk_id, chunk_detail in all_chunk_data.items():
        passage_dpr_score = normalized_dpr_scores.get(chunk_id, 0.0)
        for entity_id in chunk_detail.get('entity_ids_in_chunk', []):
            if entity_id in G_nx: # Đảm bảo entity có trong đồ thị chính
                entity_to_dpr_passage_score_sum[entity_id] += passage_dpr_score
                entity_passage_counts[entity_id] += 1

    for entity_id, total_score in entity_to_dpr_passage_score_sum.items():
        if entity_passage_counts[entity_id] > 0:
            avg_passage_score_for_entity = total_score / entity_passage_counts[entity_id]
            personalization_vector[entity_id] += avg_passage_score_for_entity * passage_influence_factor
    print("4b. Weights from DPR passage scores added to personalization vector.")

    # 4c. Synonym Expansion (Optional: tăng trọng số cho các node đồng nghĩa)
    # Với mỗi node có trọng số cao trong personalization_vector, cũng tăng trọng số cho các node đồng nghĩa của nó
    # Đây là phiên bản đơn giản, bạn có thể làm phức tạp hơn (ví dụ: chia sẻ một phần trọng số)
    if config.get('use_synonyms', False):
        nodes_with_initial_weights = [node for node, weight in personalization_vector.items() if weight > 1e-6]
        for node_id in nodes_with_initial_weights:
            original_weight = personalization_vector[node_id]
            for synonym_id in synonym_data.get(node_id, []):
                if synonym_id in personalization_vector:
                    # Ví dụ: thêm một phần trọng số của node gốc, hoặc bằng trọng số gốc
                    personalization_vector[synonym_id] += original_weight * config.get('synonym_weight_factor', 0.5)
        print("4c. Synonym expansion applied to personalization vector.")


    # Chuẩn hóa personalization_vector để tổng bằng 1 (thường là yêu cầu của PageRank)
    # Hoặc đảm bảo không có giá trị âm và có ít nhất một giá trị dương.
    current_sum = sum(personalization_vector.values())
    if current_sum > 1e-9: # Tránh chia cho 0
        personalization_vector = {k: v / current_sum for k, v in personalization_vector.items()}
    else: # Nếu tất cả trọng số là 0 (không có fact và passage nào khớp), dùng uniform
        print("Warning: Personalization vector is all zeros. Using uniform distribution.")
        if len(G_nx.nodes()) > 0:
            uniform_weight = 1.0 / len(G_nx.nodes())
            personalization_vector = {node: uniform_weight for node in G_nx.nodes()}
        else: # Đồ thị rỗng
             print("Error: Graph has no nodes.")
             return []


    # 5. Run Personalized PageRank
    alpha = 1.0 - config.get('damping_factor', 0.5) # alpha = 1 - damping
    print(f"5. Running Personalized PageRank with alpha={alpha:.2f}...")
    if not G_nx.nodes():
        print("Graph is empty, cannot run PageRank.")
        return []
    try:
        ppr_scores = nx.pagerank(
            G_nx,
            alpha=alpha,
            personalization=personalization_vector,
            weight='weight', # Sử dụng trọng số cạnh đã có của bạn
            tol=1.0e-6, # Dung sai cho sự hội tụ
            max_iter=100
        )
    except nx.PowerIterationFailedConvergence:
        print("Warning: PageRank did not converge. Using results from last iteration.")
        # max_iter có thể cần tăng, hoặc tol giảm
        # Hoặc chấp nhận kết quả không hội tụ hoàn toàn (networkx vẫn trả về kết quả gần đúng)
        # Trong trường hợp này, chúng ta cần lấy kết quả từ lần lặp cuối.
        # Tuy nhiên, nx.pagerank sẽ raise lỗi. Để đơn giản, ta có thể bỏ qua lỗi này
        # và chấp nhận là nó sẽ không hội tụ nếu dữ liệu quá phức tạp/sparse.
        # Hoặc dùng:
        ppr_scores = nx.pagerank(G_nx, alpha=alpha, personalization=personalization_vector, weight='weight', tol=1.0e-4, max_iter=500, nstart=None)


    print("PageRank scores calculated for entity nodes.")

    # 6. Map PPR Entity Scores to Chunks and Rank Chunks
    # Điểm của một chunk có thể là tổng/trung bình/max của điểm PPR của các entities trong chunk đó.
    chunk_final_scores = defaultdict(float)

    for chunk_id, chunk_detail in all_chunk_data.items():
        score_for_this_chunk = 0
        entities_in_this_chunk = chunk_detail.get('entity_ids_in_chunk', [])
        if not entities_in_this_chunk:
            # Nếu chunk không có entity nào được map, có thể cho nó điểm DPR ban đầu
            score_for_this_chunk = normalized_dpr_scores.get(chunk_id, 0.0) * config.get('dpr_only_chunk_factor', 0.1) # Giảm nhẹ
        else:
            num_entities_in_chunk_in_graph = 0
            for entity_id in entities_in_this_chunk:
                if entity_id in ppr_scores: # Entity có trong đồ thị và có điểm PPR
                    score_for_this_chunk += ppr_scores.get(entity_id, 0.0)
                    num_entities_in_chunk_in_graph +=1
            if num_entities_in_chunk_in_graph > 0 and config.get('average_ppr_for_chunk', False):
                 score_for_this_chunk /= num_entities_in_chunk_in_graph # Trung bình điểm PPR

        # Có thể thêm một phần điểm DPR trực tiếp vào chunk score
        score_for_this_chunk += normalized_dpr_scores.get(chunk_id, 0.0) * config.get('direct_dpr_to_chunk_weight', 0.05)
        chunk_final_scores[chunk_id] = score_for_this_chunk

    sorted_chunks = sorted(chunk_final_scores.items(), key=lambda item: item[1], reverse=True)
    print("6. PPR scores mapped to chunks and chunks ranked.")

    # 7. Return top N chunks
    retrieved_chunk_ids_with_scores = sorted_chunks[:config.get('num_retrieved_chunks', 5)]
    print(f"--- Retrieval Complete. Top {len(retrieved_chunk_ids_with_scores)} chunks: ---")
    for chunk_id, score in retrieved_chunk_ids_with_scores:
        print(f"Chunk ID: {chunk_id}, Score: {score:.4f}, Content: {all_chunk_data[chunk_id]['content'][:100]}...")

    return retrieved_chunk_ids_with_scores

# --- CÁCH SỬ DỤNG VÍ DỤ ---
if __name__ == '__main__':
    # 1. Load Graph của bạn
    # G_nx = nx.load_gml("your_graph.gml") # Hoặc cách bạn load đồ thị
    G_nx = nx.Graph() # Ví dụ đồ thị rỗng
    entities = ["entity_A", "entity_B", "entity_C", "entity_D", "entity_E"]
    G_nx.add_nodes_from(entities)
    G_nx.add_edge("entity_A", "entity_B", weight=0.5, chunk_ids=["chunk_1"])
    G_nx.add_edge("entity_B", "entity_C", weight=0.8, chunk_ids=["chunk_1", "chunk_2"])
    G_nx.add_edge("entity_A", "entity_C", weight=0.2, chunk_ids=["chunk_2"])
    G_nx.add_edge("entity_D", "entity_E", weight=0.9, chunk_ids=["chunk_3"])
    # Gán chunk_ids cho nodes (ví dụ)
    # for node in G_nx.nodes(): G_nx.nodes[node]['chunk_ids'] = [] # Khởi tạo
    # G_nx.nodes["entity_A"]['chunk_ids'] = ["chunk_1", "chunk_2"]
    # G_nx.nodes["entity_B"]['chunk_ids'] = ["chunk_1"]
    # G_nx.nodes["entity_C"]['chunk_ids'] = ["chunk_1", "chunk_2"]
    # G_nx.nodes["entity_D"]['chunk_ids'] = ["chunk_3"]
    # G_nx.nodes["entity_E"]['chunk_ids'] = ["chunk_3"]


    # 2. Chuẩn bị all_chunk_data
    all_chunk_data_example = {
        "chunk_1": {"content": "This is the first chunk about entity A and B, also C.", "entity_ids_in_chunk": ["entity_A", "entity_B", "entity_C"]},
        "chunk_2": {"content": "Second chunk mentions A and C.", "entity_ids_in_chunk": ["entity_A", "entity_C"]},
        "chunk_3": {"content": "The third chunk is about D and E.", "entity_ids_in_chunk": ["entity_D", "entity_E"]},
        "chunk_4": {"content": "An unrelated chunk.", "entity_ids_in_chunk": []},
    }

    # 3. Load synonym_data
    # synonym_data_example = {"entity_A": ["entity_A_synonym"], "entity_X": ["entity_Y"]}
    synonym_data_example = {} # Để trống nếu không dùng

    # 4. Cấu hình
    config_example = {
        "linking_top_k_facts": 5,       # Số fact/edge hàng đầu để xem xét
        "passage_node_weight_factor": 0.2, # Ảnh hưởng của DPR lên personalization vector của entity
        "damping_factor": 0.5,          # Damping cho PageRank (alpha = 1 - damping)
        "num_retrieved_chunks": 3,      # Số chunk trả về
        "use_synonyms": False,          # Có sử dụng synonym không
        "synonym_weight_factor": 0.5,   # Hệ số trọng số cho synonym
        "direct_dpr_to_chunk_weight": 0.1, # Thêm một phần điểm DPR trực tiếp vào điểm chunk cuối cùng
        "average_ppr_for_chunk": True, # Nếu true, điểm chunk là trung bình PPR của entities, nếu false là tổng
        "dpr_only_chunk_factor": 0.01,  # Trọng số cho chunk chỉ có điểm DPR (không có entity liên quan PPR)

    }

    # 5. Câu truy vấn
    test_query = "Tell me about A and B"

    # 6. Chạy retrieval
    retrieved_results = reproduce_hipporag_retrieval_nx(
        G_nx,
        test_query,
        all_chunk_data_example,
        synonym_data_example,
        config_example
    )