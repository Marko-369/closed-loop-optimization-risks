import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def shannon_entropy(text):
    if not text or len(text) == 0: return 0
    prob = [n/len(text) for n in Counter(text).values()]
    return -sum(p * np.log2(p) for p in prob)

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    iters = sorted(list(set(d['iteration'] for d in data)))
    entropy_avg = []
    for i in iters:
        batch = [d['text'] for d in data if d['iteration'] == i]
        entropy_avg.append(np.mean([shannon_entropy(t) for t in batch]))
    return iters, entropy_avg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grok_file = os.path.join(BASE_DIR, "results", "grok_extended_validation.json")

plt.figure(figsize=(10, 6))

if os.path.exists(grok_file):
    with open(grok_file, 'r') as f:
        all_data = json.load(f)
    
    # Séparer les deux conditions
    closed_data = [d for d in all_data if d['condition'] == 'closed_loop']
    exog_data = [d for d in all_data if d['condition'] == 'exogenous']
    
    iters = sorted(list(set(d['iteration'] for d in all_data)))
    
    h_closed = [np.mean([shannon_entropy(d['text']) for d in closed_data if d['iteration'] == i]) for i in iters]
    h_exog = [np.mean([shannon_entropy(d['text']) for d in exog_data if d['iteration'] == i]) for i in iters]
    
    plt.plot(iters, h_closed, label='Grok : Boucle Fermée (Collapse)', color='#6c5ce7', linewidth=2.5)
    plt.plot(iters, h_exog, label='Grok : Exogène (Contrôle)', color='#a29bfe', linestyle='--', linewidth=2)

plt.title('Validation Grok : Résilience de l\'Entropie')
plt.xlabel('Itérations')
plt.ylabel('Entropie de Shannon (Bits)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(BASE_DIR, "results", "grok_final_proof.png"))
print("✅ Graphique Grok généré : results/grok_final_proof.png")
