from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

intent_examples = {

    "nec": [
        "electrical code rules",
        "grounding requirements",
        "ampacity definition",
        "overcurrent protection",
        "nec wiring rules",
        "national electrical code",
        "article section",
        "conductor sizing",
        "circuit protection"
    ],

    "solar": [
        "solar panel installation",
        "pv module wiring",
        "solar inverter connection",
        "photovoltaic system setup",
        "solar array configuration",
        "solar panel installation guide",
        "solar system design",
        "pv system commissioning",
        "solar equipment specifications"
    ],

    "wattmonk": [
        "what does wattmonk do",
        "wattmonk services",
        "company information",
        "who founded wattmonk",
        "wattmonk business model",
        "wattmonk experience",
        "wattmonk expertise",
        "wattmonk capabilities",
        "wattmonk offerings"
    ]
}

intent_embeddings = {
    intent: model.encode(samples)
    for intent, samples in intent_examples.items()
}

CONFIDENCE_THRESHOLD = 0.3


def route_query(query):
    """
    Route query to appropriate intent based on semantic similarity.
    
    Args:
        query (str): User query string
        
    Returns:
        str: Intent classification ('nec', 'solar', 'wattmonk')
    """
    query_embedding = model.encode(query)

    scores = {}

    for intent, embeddings in intent_embeddings.items():

        similarity = np.dot(embeddings, query_embedding)
        scores[intent] = similarity.mean()

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    
    

    return best_intent