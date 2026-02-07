import json
from typing import List, Dict, Any
from rag.logging_utils import get_logger

logger = get_logger(__name__)

def load_dataset(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Dataset JSON is not a list. Found: %s", type(data))
            return []
        logger.info("Loaded dataset: %d docs", len(data))
        return data
    except Exception as e:
        logger.warning("⚠️ Dataset not found/invalid (%s): %s", path, e)
        return []
