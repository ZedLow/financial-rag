import gradio as gr

def build_demo(retrieve_and_answer):
    with gr.Blocks(title="Financial RAG Demo") as demo:
        gr.Markdown("# Financial RAG Demo")
        gr.Markdown(
            "Architecture: GLiNER (CPU Router) → GTE-7B (Retrieval) → Qwen2-VL (Vision Analysis)"
        )

        gr.Markdown(
    """
### About this demo
- Entity-aware routing (Apple / Microsoft)
- Prevents *accidental* cross-company document mixing
- Tables processed as images (no OCR flattening)
- Explicit refusals for out-of-scope or ambiguous queries

### Try these prompts
- `What was Apple's total revenue in 2023?`
- `What is Microsoft's operating income?`
- `Compare Apple and Microsoft revenues` -> explicit multi-entity reasoning
- `What was Google's revenue in 2023?` -> rejected

i Full details are in the GitHub repo.
"""
)


        with gr.Row():
            query_input = gr.Textbox(
                label="User Question",
                placeholder=(
                    "Example: Compare Apple and Microsoft revenue, "
                    "or ask about a specific company metric."
                ),
            )
            submit_btn = gr.Button("Run Analysis", variant="primary")

        with gr.Row():
            output_gallery = gr.Gallery(label="Source Documents", columns=3, height=300)
            output_meta = gr.Markdown(label="System Trace")
            output_text = gr.Markdown(label="Answer")

        submit_btn.click(
            retrieve_and_answer,
            inputs=query_input,
            outputs=[output_gallery, output_meta, output_text],
        )

    return demo
