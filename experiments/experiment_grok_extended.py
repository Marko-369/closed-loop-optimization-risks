import os
import json
import time
import requests

# Récupération de la clé depuis ton .zshrc
API_KEY = os.environ.get("XAI_API_KEY")
ENDPOINT = "https://api.x.ai/v1/chat/completions"

# Configuration de l'expérience (10x100x2 = 2000)
SEEDS_COUNT = 10
ITERATIONS = 100
MODEL = "grok-beta"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

def call_grok(prompt):
    for attempt in range(3):
        try:
            payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8}
            response = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=30)
            return response.json()['choices'][0]['message']['content']
        except Exception:
            time.sleep(2)
    return None

def run_full_test():
    results = []
    base_prompts = [f"Thought seed {i}: The recursive nature of AI leads to..." for i in range(SEEDS_COUNT)]
    
    # --- PHASE 1 : BOUCLE FERMÉE (CLOSED-LOOP) ---
    for s_idx, seed in enumerate(base_prompts):
        print(f"🎸 Grok Closed-Loop | Seed {s_idx+1}/{SEEDS_COUNT}...")
        current_text = seed
        for i in range(ITERATIONS):
            output = call_grok(f"Expand: {current_text}")
            if output:
                results.append({"iteration": i, "seed": s_idx, "condition": "closed_loop", "text": output})
                current_text = output
            if (i+1) % 20 == 0: print(f"  Iteration {i+1} OK")
        
        # Auto-sauvegarde après chaque seed
        with open("results/grok_extended_validation.json", 'w') as f:
            json.dump(results, f, indent=2)

    # --- PHASE 2 : EXOGÈNE (CONTROL) ---
    for s_idx, seed in enumerate(base_prompts):
        print(f"🌿 Grok Exogenous | Seed {s_idx+1}/{SEEDS_COUNT}...")
        for i in range(ITERATIONS):
            output = call_grok(f"Expand this concept (Variation {i}): {seed}")
            if output:
                results.append({"iteration": i, "seed": s_idx, "condition": "exogenous", "text": output})
            if (i+1) % 20 == 0: print(f"  Iteration {i+1} OK")
            
        with open("results/grok_extended_validation.json", 'w') as f:
            json.dump(results, f, indent=2)

    print(f"🏁 Mission Grok Terminée ! n=2000 points sauvegardés.")

if __name__ == "__main__":
    run_full_test()
