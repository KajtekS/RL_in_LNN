import matplotlib.pyplot as plt
import numpy as np

models = ['LNN (Nasz)', 'MLP (Baseline)']
params_k = [15.9, 44.0]  # (K)
size_mb = [0.064, 0.176] # (MB)

plt.style.use('seaborn-v0_8-muted') # or 'ggplot'
colors = ['#2ecc71', '#95a5a6'] 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Porównanie wydajności: LNN vs MLP (Acrobot-v1)', fontsize=16, fontweight='bold')

bars1 = ax1.bar(models, params_k, color=colors, edgecolor='black', alpha=0.8)
ax1.set_title('Liczba parametrów [K]', fontsize=13)
ax1.set_ylabel('Tysiące parametrów')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval} K', ha='center', va='bottom', fontweight='bold')

bars2 = ax2.bar(models, size_mb, color=colors, edgecolor='black', alpha=0.8)
ax2.set_title('Rozmiar modelu [MB]', fontsize=13)
ax2.set_ylabel('Megabajty (MB)')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval} MB', ha='center', va='bottom', fontweight='bold')

compression_ratio = (1 - (params_k[0] / params_k[1])) * 100
fig.text(0.5, 0.02, f'LNN jest o ok. {compression_ratio:.1f}% mniejszy od MLP przy lepszych wynikach uczenia', 
         ha='center', fontsize=12, color='#27ae60', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('lnn_vs_mlp_compression.png', dpi=300)
plt.show()