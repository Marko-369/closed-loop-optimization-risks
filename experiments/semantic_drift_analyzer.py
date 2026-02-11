import numpy as np
from anthropic import Anthropic
from scipy.spatial.distance import cosine

# Configuration
client = Anthropic() # Utilise ta clé exportée en variable d'environnement

def get_embedding(text):
    # Note : On utilise un modèle d'embedding pour transformer le texte en vecteur
    # (Par exemple via OpenAI ou un modèle local comme Sentence-Transformers)
    # Pour rester pur Anthropic, on peut analyser la variation des probabilités
    pass

def analyze_semantic_drift(generations):
    """
    Calcule la distance sémantique entre l'itération N et N+1.
    Une distance qui rétrécit confirme l'effondrement de l'exploration.
    """
    distances = []
    for i in range(len(generations) - 1):
        vec_a = generations[i]['vector']
        vec_b = generations[i+1]['vector']
        # Distance cosinus : 0 = identique, 1 = orthogonal
        dist = cosine(vec_a, vec_b)
        distances.append(dist)
    return distances