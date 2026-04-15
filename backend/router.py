from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

intent_examples = {

    "skin_cancer": [
        "skin cancer types",
        "melanoma symptoms",
        "basal cell carcinoma",
        "squamous cell carcinoma",
        "skin cancer detection",
        "skin cancer risk factors",
        "skin cancer prevention",
        "skin cancer treatment",
        "skin lesion characteristics",
        "ABCDE rule",
        "skin cancer screening",
        "dermatology"
    ],

    "general_health_faq": [
        "skin health",
        "sun protection",
        "sunscreen recommendations",
        "skin care tips",
        "dermatologist visit",
        "skin disease",
        "skin condition",
        "health FAQ",
        "general medical question"
    ]
}

intent_embeddings = {
    intent: model.encode(samples)
    for intent, samples in intent_examples.items()
}

CONFIDENCE_THRESHOLD = 0.25


def route_query(query):
    """
    Route query to appropriate intent based on semantic similarity.
    
    Args:
        query (str): User query string
        
    Returns:
        str: Intent classification ('skin_cancer', 'general_health_faq')
    """
    query_embedding = model.encode(query)

    scores = {}

    for intent, embeddings in intent_embeddings.items():

        similarity = np.dot(embeddings, query_embedding)
        scores[intent] = similarity.mean()

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    
    # If confidence is too low, route to general health FAQ as fallback
    if best_score < CONFIDENCE_THRESHOLD:
        best_intent = "general_health_faq"

    return best_intent