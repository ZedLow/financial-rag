import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from rag.config import Settings
from rag.data import Doc
from rag.logging_utils import get_logger

logger = get_logger(__name__)

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def _fingerprint(docs: List[Doc], settings: Settings) -> str:
    h = hashlib.sha256()
    h.update(settings.embed_model_id.encode("utf-8"))
    h.update(str(settings.embed_max_len).encode("utf-8"))
    for d in docs:
        h.update(d.doc_name.encode("utf-8"))
        h.update(d.company.encode("utf-8"))
        h.update(d.text.encode("utf-8"))
    return h.hexdigest()

def ensure_index_dir(settings: Settings):
    Path(settings.index_dir).mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def build_or_load_doc_embeddings(
    docs: List[Doc],
    embed_tokenizer,
    embed_model,
    settings: Settings,
) -> Tuple[torch.Tensor, str]:
    """
    Returns (doc_embeddings [N, D] on CPU, fingerprint)
    Caches to data/index/doc_embeds.pt
    """
    ensure_index_dir(settings)
    fp = _fingerprint(docs, settings)
    cache_file = settings.doc_embeds_file()
    meta_file = settings.doc_meta_file()

    if cache_file.exists() and meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fp:
                logger.info("Loading cached doc embeddings: %s", str(cache_file))
                payload = torch.load(cache_file, map_location="cpu")
                return payload["embeddings"], fp
        except Exception as e:
            logger.warning("Failed to load cache, rebuilding. Reason: %s", e)

    logger.info("Building doc embeddings cache (%d docs)...", len(docs))
    doc_texts = [d.text for d in docs]
    embs = []

    for i in range(0, len(doc_texts), settings.embed_batch_size):
        batch = doc_texts[i : i + settings.embed_batch_size]
        d_inputs = embed_tokenizer(
            batch,
            max_length=settings.embed_max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(embed_model.device)

        d_outputs = embed_model(**d_inputs)
        batch_emb = last_token_pool(d_outputs.last_hidden_state, d_inputs["attention_mask"])
        batch_emb = F.normalize(batch_emb, p=2, dim=1)
        embs.append(batch_emb.detach().to("cpu"))

    doc_embs = torch.cat(embs, dim=0)

    torch.save({"embeddings": doc_embs}, cache_file)
    meta_file.write_text(json.dumps({"fingerprint": fp, "n_docs": len(docs)}, indent=2), encoding="utf-8")
    logger.info("Saved embeddings cache: %s", str(cache_file))
    return doc_embs, fp

@torch.no_grad()
def embed_query(query: str, embed_tokenizer, embed_model, settings: Settings) -> torch.Tensor:
    query_text = (
        "Instruct: Given a user query, retrieve relevant passages that answer the query.\n"
        f"Query: {query}"
    )
    q_inputs = embed_tokenizer(
        [query_text],
        max_length=settings.embed_max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(embed_model.device)

    q_outputs = embed_model(**q_inputs)
    q_emb = last_token_pool(q_outputs.last_hidden_state, q_inputs["attention_mask"])
    q_emb = F.normalize(q_emb, p=2, dim=1)
    return q_emb.detach().to("cpu")  # keep retrieval ops on CPU

def topk_retrieval(q_emb_cpu: torch.Tensor, doc_embs_cpu: torch.Tensor, k: int) -> List[int]:
    # q_emb: [1, D], doc_embs: [N, D]
    scores = (q_emb_cpu @ doc_embs_cpu.T).squeeze(0)
    k = min(k, scores.shape[0])
    return torch.topk(scores, k=k).indices.tolist()

@torch.no_grad()
def rerank(
    query: str,
    candidate_docs: List[Doc],
    rerank_tokenizer,
    rerank_model,
    settings: Settings,
    k: int,
) -> Tuple[List[int], torch.Tensor]:
    pairs = [[query, d.text] for d in candidate_docs]
    r_inputs = rerank_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=settings.rerank_max_len,
    ).to(rerank_model.device)

    r_scores = rerank_model(**r_inputs, return_dict=True).logits.view(-1).float().detach().to("cpu")
    k = min(k, len(candidate_docs))
    top_idx = torch.topk(r_scores, k=k).indices.tolist()
    return top_idx, r_scores
