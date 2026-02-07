import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Any, Callable, Tuple
from rag.prompting import build_messages
from rag.config import Settings
from rag.logging_utils import get_logger
from rag.utils import last_token_pool
from qwen_vl_utils import process_vision_info

logger = get_logger(__name__)

def _route_companies(
    query: str,
    router_model,
    settings: Settings,
) -> Tuple[List[str], str | None]:
    
    allowed_companies = {
        "apple": "Apple", 
        "aapl": "Apple",
        "microsoft": "Microsoft", 
        "msft": "Microsoft"
    }

    labels = list(settings.router_labels)
    entities = router_model.predict_entities(query, labels, threshold=settings.router_threshold)
    
    detected_targets = []
    unsupported_targets = []
    
    for e in entities:
        name_clean = (e.get("text") or "").lower().strip()
        found_match = False
        
        for key, canonical_name in allowed_companies.items():
            if key in name_clean:
                detected_targets.append(canonical_name)
                found_match = True
                break
        
        if not found_match:
            unsupported_targets.append(e.get("text"))

    detected_targets = list(set(detected_targets))
    unsupported_targets = list(set(unsupported_targets))

    if unsupported_targets:
        return [], (
            f"⛔ **Out of Scope:** I detected a request for **{', '.join(unsupported_targets)}**. "
            "This system only has access to **Microsoft** and **Apple** data."
        )

    if not detected_targets:
        return [], (
            "❓ **Ambiguous Query:** I could not identify a specific company (Apple or Microsoft). "
            "Please name the company you want to analyze."
        )

    return detected_targets, None

def _filter_docs(
    dataset: List[Dict[str, Any]],
    detected_companies: List[str],
) -> List[Dict[str, Any]]:
    
    valid_docs = []
    for i, doc in enumerate(dataset):
        doc_name = doc.get("doc_name", "Doc")
        
        if detected_companies:
            if not any(company in doc_name for company in detected_companies):
                continue
        
        text = (doc.get("text") or "").strip()
        if text:
            valid_docs.append({"text": text, "original_index": i, "doc_name": doc_name})
            
    return valid_docs

def _prepare_images(
    dataset: List[Dict[str, Any]],
    valid_docs: List[Dict[str, Any]],
    top_k_indices: List[int],
    r_scores,
    top_k_indices_local: List[int],
):
    images_content = []
    gallery_preview = []
    meta_info = ""
    
    for idx_local in top_k_indices_local:
        idx_in_valid = top_k_indices[idx_local]
        final_doc_idx = valid_docs[idx_in_valid]["original_index"]
        
        doc = dataset[final_doc_idx]
        image_path = doc["image_path"]
        score = r_scores[idx_local].item()
        doc_name = doc.get("doc_name", "Unknown")
        
        try:
            img = Image.open(image_path)
            header_text = f"SOURCE DOCUMENT: {doc_name} (Confidence: {score:.2f})\n"
            
            images_content.append({"type": "text", "text": header_text})
            images_content.append({"type": "image", "image": img})
            
            gallery_preview.append((img, doc_name))
            meta_info += f"- **{doc_name}** (Score: {score:.2f})\n"
        except Exception as e:
            logger.warning("Failed to open image %s: %s", image_path, e)
            continue
            
    return images_content, gallery_preview, meta_info

def make_retrieve_and_answer(
    dataset: List[Dict[str, Any]],
    models,
    settings: Settings | None = None,
) -> Callable[[str], tuple]:
   
    if settings is None:
        settings = models.settings if hasattr(models, "settings") else Settings()

    import spaces

    @spaces.GPU
    def retrieve_and_answer(query: str):
        logger.info("User question: %s", query)
        
        if not dataset:
            return [], "Empty corpus", "No documents loaded."

        detected_companies, blocked_msg = _route_companies(query, models.router_model, settings)
        
        if blocked_msg is not None:
            return [], "", blocked_msg
            
        logger.info("Router detected companies: %s", detected_companies)

        valid_docs = _filter_docs(dataset, detected_companies)
        
        if not valid_docs:
            return [], "", "System Error: Valid targets detected but no matching documents found."

        query_text = (
            "Instruct: Given a user query, retrieve relevant passages that answer the query.\n"
            f"Query: {query}"
        )
        
        with torch.no_grad():
            q_inputs = models.embed_tokenizer(
                [query_text],
                max_length=settings.embed_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(models.embed_model.device)
            
            q_outputs = models.embed_model(**q_inputs)
            q_emb = last_token_pool(q_outputs.last_hidden_state, q_inputs["attention_mask"])
            q_emb = F.normalize(q_emb, p=2, dim=1)
            
            d_embeddings_list = []
            doc_texts = [d["text"] for d in valid_docs]
            
            for i in range(0, len(doc_texts), 1):
                d_inputs = models.embed_tokenizer(
                    doc_texts[i:i + 1],
                    max_length=settings.embed_max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(models.embed_model.device)
                
                d_outputs = models.embed_model(**d_inputs)
                batch_emb = last_token_pool(d_outputs.last_hidden_state, d_inputs["attention_mask"])
                batch_emb = F.normalize(batch_emb, p=2, dim=1)
                d_embeddings_list.append(batch_emb)
                
            d_emb_final = torch.cat(d_embeddings_list, dim=0)
            scores = (q_emb @ d_emb_final.T).squeeze(0)
            
            k_val = min(settings.embed_top_k, len(scores))
            top_k_indices = torch.topk(scores, k=k_val).indices.tolist()

        pairs = [[query, valid_docs[idx]["text"]] for idx in top_k_indices]
        
        with torch.no_grad():
            r_inputs = models.rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=settings.rerank_max_length,
            ).to(models.rerank_model.device)
            
            r_scores = models.rerank_model(**r_inputs, return_dict=True).logits.view(-1).float()
            
            k_rerank = min(settings.rerank_top_k, len(r_scores))
            top_k_indices_local = torch.topk(r_scores, k=k_rerank).indices.tolist()

        meta_info = f"**AI Router Focus:** {', '.join(detected_companies)}\n\n"
        
        images_content, gallery_preview, meta_sources = _prepare_images(
            dataset, valid_docs, top_k_indices, r_scores, top_k_indices_local
        )
        meta_info += meta_sources
        
        if not images_content:
            return [], "", "No images found for the retrieved passages."

        messages = build_messages(query, images_content)
        
        text_input = models.gen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _video_inputs = process_vision_info(messages)
        
        inputs = models.gen_processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(models.gen_model.device)
        
        generated_ids = models.gen_model.generate(**inputs, max_new_tokens=settings.max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = models.gen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return gallery_preview, meta_info, response

    return retrieve_and_answer