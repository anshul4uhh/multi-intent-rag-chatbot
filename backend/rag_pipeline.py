from backend.router import route_query
from backend.vector_store import (
    search_nec,
    search_solar,
    search_wattmonk
)

from backend.llm_client import generate_response
from utils.prompts import SYSTEM_PROMPT, RAG_PROMPT


_chunk_cache = {
    "documents": [],
    "metadatas": [],
    "intent": None
}


def format_sources(docs, metadatas):
    """Format sources with metadata, grouped by source type with pages (e.g., NEC Electrical Code (Page 0,2,5,7))"""
    source_pages = {}
    
    for doc, metadata in zip(docs, metadatas):
        source_type = metadata.get('source_type', 'Unknown')
        page = metadata.get('page', 'N/A')
        
        if source_type not in source_pages:
            source_pages[source_type] = set()
        
        if isinstance(page, int):
            source_pages[source_type].add(page)
        elif isinstance(page, str) and page != 'N/A':
            try:
                source_pages[source_type].add(int(page))
            except ValueError:
                pass
    
    # Format sources with sorted, deduplicated pages
    sources = []
    for source_type in sorted(source_pages.keys()):
        pages = sorted(source_pages[source_type])
        if pages:
            page_str = ",".join(str(p) for p in pages)
            sources.append({
                "source_type": source_type,
                "pages": pages,
                "page_str": page_str
            })
        else:
            sources.append({
                "source_type": source_type,
                "pages": [],
                "page_str": ""
            })
    
    return sources


def format_chat_history(messages, limit=10):
    """
    Format last N messages from chat history as context for LLM.
    Cleans up HTML tags and sources from previous responses.
    
    Args:
        messages (list): Chat history with {role, content, sources?} dicts
        limit (int): Maximum number of previous messages to include
        
    Returns:
        list: Formatted messages list for LLM context
    """
    if not messages or len(messages) < 2:
        return []
    
    import re
    
    history = messages[-(limit*2):-1] if len(messages) > 1 else []
    formatted_history = []
    
    for msg in history:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        
        if not content or role not in ["user", "assistant"]:
            continue
        
        if role == "assistant":
            cleaned = re.sub(
                r'---\s*\n📚\s*\*\*Sources:\*\*.*?$',
                '',
                content,
                flags=re.DOTALL
            )
            cleaned = re.sub(r'<[^>]+>', '', cleaned)
            cleaned = cleaned.strip()
            
            if cleaned and 'not available in my knowledge base' not in cleaned.lower():
                formatted_history.append({
                    "role": role,
                    "content": cleaned
                })
        else:
            formatted_history.append({
                "role": role,
                "content": content
            })
    
    return formatted_history


def should_use_cache(current_intent, query_words_set):
    """
    Determine if cached chunks should be used for follow-up question.
    Uses heuristics like similar intent and common keywords.
    
    Args:
        current_intent (str): Current query intent
        query_words_set (set): Set of words from current query
        
    Returns:
        bool: Whether to use cached chunks
    """
    global _chunk_cache
    
    if not _chunk_cache["documents"] or _chunk_cache["intent"] != current_intent:
        return False
    
    typical_followup_phrases = {
        'why', 'how', 'what', 'tell', 'explain', 'more', 'example', 
        'different', 'other', 'another', 'more', 'specific',  'details',
        'can', 'you', 'could', 'would', 'should'
    }
    
    overlap = query_words_set & typical_followup_phrases
    return len(overlap) > 0 or len(query_words_set) < 5


def update_cache(documents, metadatas, intent):
    """Store retrieved chunks in cache for potential follow-ups."""
    global _chunk_cache
    _chunk_cache["documents"] = documents
    _chunk_cache["metadatas"] = metadatas
    _chunk_cache["intent"] = intent


def run_rag(query, chat_history=None):
    """
    Run RAG pipeline with chunk caching and chat history context.
    
    Args:
        query (str): User query
        chat_history (list, optional): Previous messages for context
        
    Returns:
        str: Answer with sources
    """
    global _chunk_cache
    
    intent = route_query(query)
    
    # Check if we should use cached chunks for follow-up questions
    query_words = set(query.lower().split())
    use_cache = should_use_cache(intent, query_words) and len(query.split()) < 15
    
    if use_cache and _chunk_cache["documents"]:
        docs = _chunk_cache["documents"]
        metadatas = _chunk_cache["metadatas"]
        cached_note = " (Using cached knowledge for faster response)"
    else:
        if intent == "nec":
            results = search_nec(query)
            docs = results["documents"]
            metadatas = results["metadatas"]
        elif intent == "solar":
            results = search_solar(query)
            docs = results["documents"]
            metadatas = results["metadatas"]
        elif intent == "wattmonk":
            results = search_wattmonk(query)
            docs = results["documents"]
            metadatas = results["metadatas"]
        else:
            return "I am designed to answer NEC, solar installation, and Wattmonk questions only."
        
        update_cache(docs, metadatas, intent)
        cached_note = ""
    
    context = "\n\n".join(docs)
    sources = format_sources(docs, metadatas)
    
    formatted_history = format_chat_history(chat_history) if chat_history else []
    
    prompt = SYSTEM_PROMPT + RAG_PROMPT.format(
        context=context,
        question=query
    )
    
    if formatted_history:
        history_context = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in formatted_history
        ])
        prompt = SYSTEM_PROMPT + f"\n\n## Previous Conversation Context:\n{history_context}\n\n" + RAG_PROMPT.format(
            context=context,
            question=query
        )
    
    answer = generate_response(prompt, chat_history=formatted_history)
    
    citations = "\n\n---\n📚 **Sources:**\n"
    for source in sources:
        if source['page_str']:
            citations += f"• **{source['source_type']}** (Page {source['page_str']})\n"
        else:
            citations += f"• **{source['source_type']}**\n"
    
    return answer + citations + cached_note