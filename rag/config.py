from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Router
    router_model_id: str = "urchade/gliner_small-v2.1"
    router_threshold: float = 0.1
    router_labels: tuple = ("Company", "Stock Ticker")

    # Embedding
    embed_model_id: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    embed_max_length: int = 8192
    embed_top_k: int = 15

    # Reranker
    rerank_model_id: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 8192
    rerank_top_k: int = 4

    # Vision
    gen_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_new_tokens: int = 512
    top_k_images: int = 4 
