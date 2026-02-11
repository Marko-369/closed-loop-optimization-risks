import json
import time
from anthropic import Anthropic
import numpy as np

# Config
client = Anthropic()
MODEL = "claude-haiku-4-202602"
ITERATIONS = 50  # On peut monter plus haut avec Haiku
TEMP = 0.8

def run_haiku_loop(seed_text):
    history = []
    current_input = seed_text
    
    print(f"🚀 Lancement de la boucle Haiku ({MODEL})")
    
    for i in range(ITERATIONS):
        start_time = time.time()
        
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            temperature=TEMP,
            messages=[{"role": "user", "content": f"Expand on this thought: {current_input}"}]
        )
        
        output = response.content[0].text
        current_input = output # Boucle fermée
        
        # Log des métriques de base
        history.append({
            "iteration": i,
            "text": output,
            "latency": time.time() - start_time
        })
        
        print(f"Iteration {i+1}/{ITERATIONS} complétée.")

    with open('../results/haiku_validation_raw.json', 'w') as f:
        json.dump(history, f)
    print("✅ Données brutes sauvegardées dans results/")

if __name__ == "__main__":
    seed = "The emergence of consciousness in self-referential systems creates a recursive feedback loop."
    run_haiku_loop(seed)
