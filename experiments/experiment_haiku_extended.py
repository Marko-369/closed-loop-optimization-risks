import json
import time
import os
from anthropic import Anthropic

# Initialisation (utilise ta clé exportée dans le Terminal)
client = Anthropic()

# Gestion robuste des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FILE_PATH = os.path.join(RESULTS_DIR, "haiku_extended_validation.json")

MODEL = "claude-3-haiku-20240307"
SEEDS = 10
ITERATIONS = 100
TEMP = 0.8

def run_validation():
    results = []
    seeds = [f"Thought seed {i}: The recursive nature of AI leads to..." for i in range(SEEDS)]
    
    # Créer le dossier results s'il n'existe pas
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    try:
        for seed_idx, base_seed in enumerate(seeds):
            print(f"--- Processing Seed {seed_idx+1}/{SEEDS} ---")
            current_text = base_seed
            
            for i in range(ITERATIONS):
                response = client.messages.create(
                    model=MODEL, max_tokens=256, temperature=TEMP,
                    messages=[{"role": "user", "content": f"Expand: {current_text}"}]
                )
                output = response.content[0].text
                results.append({
                    "iteration": i, "seed": seed_idx, 
                    "condition": "closed_loop", "text": output
                })
                current_text = output
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i+1} OK")
            
            # Sauvegarde intermédiaire après chaque seed
            with open(FILE_PATH, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Seed {seed_idx+1} sauvegardé.")

        print(f"🏁 Validation terminée ! Fichier disponible : {FILE_PATH}")

    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    run_validation()
