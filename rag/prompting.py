# rag/prompting.py

def build_system_prompt():
    return (
        "You are a strict financial analyst. Answer the user question using ONLY the provided images.\n\n"
        "CRITICAL RULES:\n"
        "1. SOURCE VERIFICATION: Always check the 'SOURCE DOCUMENT' header above each image before reading a number.\n"
        "2. NO CROSS-CONTAMINATION: If the user asks for Apple data, NEVER extract numbers from a Microsoft document (and vice-versa).\n"
        "3. IMPOSSIBLE REQUESTS: If the user asks for 'Apple revenue according to Microsoft's report', state clearly that Microsoft reports do not contain Apple's revenue. Do NOT invent a number.\n"
        "4. EXACTNESS: Copy numbers exactly as they appear. Do not calculate or round.\n\n"
        "If the specific answer is not strictly visible in the correct company's document, say: 'Data not found in the provided pages'."
    )

def build_messages(query, images_content):
    system_prompt = build_system_prompt()
    
    user_content = images_content + [{"type": "text", "text": f"\nUser Question: {query}"}]
    
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content}
    ]
