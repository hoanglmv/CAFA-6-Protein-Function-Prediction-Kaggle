import networkx as nx
from node2vec import Node2Vec
import numpy as np

def generate_node2vec_embeddings(obo_graph, dimensions=64, walk_length=30, num_walks=100, workers=4):
    """
    Tạo embedding cho các node trong đồ thị GO sử dụng thuật toán Node2Vec.
    
    Args:
        obo_graph: Đồ thị NetworkX (MultiDiGraph) load từ obonet.
        dimensions: Kích thước vector embedding (ví dụ: 64, 128).
        walk_length: Độ dài của mỗi bước đi ngẫu nhiên.
        num_walks: Số lượng bước đi ngẫu nhiên từ mỗi node.
        workers: Số luồng CPU sử dụng.
        
    Returns:
        dict: Dictionary ánh xạ {GO_ID: numpy_array_embedding}
    """
    print(f"🔄 Đang chuẩn bị đồ thị cho Node2Vec ({len(obo_graph.nodes)} nodes)...")
    
    # Node2Vec yêu cầu đồ thị DiGraph hoặc Graph, obonet trả về MultiDiGraph
    # Chúng ta chuyển sang DiGraph để loại bỏ các cạnh song song (nếu có)
    G = nx.DiGraph(obo_graph)

    # Khởi tạo mô hình Node2Vec
    # p=1, q=1 tương đương với DeepWalk (duyệt ngẫu nhiên không thiên kiến)
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, 
                        num_walks=num_walks, workers=workers, p=1, q=1, quiet=False)

    print("🏃 Đang huấn luyện mô hình Node2Vec (có thể mất vài phút)...")
    # Huấn luyện mô hình Word2Vec trên các bước đi ngẫu nhiên
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Trích xuất embedding
    embeddings = {}
    print("✅ Đã tạo xong Node Embeddings.")
    
    # Trả về dict {node_id: vector}
    # model.wv là KeyedVectors của gensim
    for node in G.nodes():
        if node in model.wv:
            embeddings[node] = model.wv[node]
        else:
            embeddings[node] = np.zeros(dimensions, dtype=np.float32)
            
    return embeddings