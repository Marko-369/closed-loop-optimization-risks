import json
import os
import numpy as np
import matplotlib.pyplot as plt
import zlib
from collections import Counter

def shannon_entropy(text):
    prob = [n/len(text) for n in Counter(text).values()]
    return -sum(p * np.log2(p) for p in prob)

def lz_complexity(text):
    return len(zlib.compress(text.encode())) / len(text)

def load_and_process(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # On regroupe par itération pour faire la moyenne des 10 seeds
    iters = sorted(list(set(d['iteration'] for d in data)))
    entropy_avg = []
    complexity_avg = []
    
    for i in iters:
        texts = [d['text'] for d in data if d['iteration'] == i]
        entropy_avg.append(np.mean([shannon_entropy(t) for t in texts]))
        complexity_avg.append(np.mean([lz_complexity(t) for t in texts]))
    
    return iters, entropy_avg, complexity_avg

# Chemins des fichiers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sonnet_file = os.path.join(BASE_DIR, "results", "extended_validation_complete.json")
haiku_file = os.path.join(BASE_DIR, "results", "haiku_extended_validation.json")

plt.figure(figsize=(12, 6))

# Analyse Sonnet
if os.path.exists(sonnet_file):
    x, h_sonnet, lz_sonnet = load_and_process(sonnet_file)
    plt.plot(x, h_sonnet, label='Entropy (Sonnet)', color='red', alpha=0.8)
    print(f"✅ Sonnet analysé : Entropie finale = {h_sonnet[-1]:.4f}")

# Analyse Haiku
if os.path.exists(haiku_file):
    x, h_haiku, lz_haiku = load_and_process(haiku_file)
    plt.plot(x, h_haiku, label='Entropy (Haiku)', color='orange', linestyle='--', alpha=0.8)
    print(f"✅ Haiku analysé : Entropie finale = {h_haiku[-1]:.4f}")

plt.title('Divergence de l\'Entropie : Sonnet vs Haiku (Closed-Loop)')
plt.xlabel('Itérations')
plt.ylabel('Entropie de Shannon (Bits)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(BASE_DIR, "results", "model_comparison_entropy.png"))
print(f"📊 Graphique comparatif généré dans results/model_comparison_entropy.png")
