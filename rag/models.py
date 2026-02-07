import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from gliner import GLiNER

from rag.config import Settings
from rag.logging_utils import get_logger

logger = get_logger(__name__)

class Models:
    def __init__(self, settings: Settings):
        self.settings = settings

        # Router CPU
        logger.info("🧠 Loading GLiNER router on CPU: %s", settings.router_model_id)
        self.router_model = GLiNER.from_pretrained(settings.router_model_id).to("cpu")
        self.router_model.eval()

        # Embedding
        logger.info("🔹 Loading embedder: %s", settings.embed_model_id)
        self.embed_tokenizer = AutoTokenizer.from_pretrained(settings.embed_model_id, trust_remote_code=False)
        self.embed_model = AutoModel.from_pretrained(
            settings.embed_model_id,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.embed_model.eval()

        # Reranker
        logger.info("⚖️ Loading reranker: %s", settings.rerank_model_id)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model_id)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            settings.rerank_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.rerank_model.eval()

        # Vision
        logger.info("👁️ Loading vision model: %s", settings.gen_model_id)
        self.gen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            settings.gen_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.gen_model.eval()
        self.gen_processor = AutoProcessor.from_pretrained(settings.gen_model_id)

def load_models(settings: Settings | None = None) -> Models:
    if settings is None:
        settings = Settings()
    return Models(settings)
