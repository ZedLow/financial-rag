import spaces  # MUST be first on HF Spaces / ZeroGPU

from rag.logging_utils import get_logger
from rag.models import load_models
from rag.data import load_dataset
from rag.pipeline import make_retrieve_and_answer
from rag.ui import build_demo

logger = get_logger(__name__)
logger.info("🚀 Starting RAG Finance Pro (Agentic Routing with GLiNER)")

dataset = load_dataset("data/dataset.json")
models = load_models()

retrieve_and_answer = make_retrieve_and_answer(dataset, models)
demo = build_demo(retrieve_and_answer)

if __name__ == "__main__":
    demo.launch()
