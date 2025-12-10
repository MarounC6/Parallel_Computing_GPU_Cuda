import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

executables = ['pi_sequential', 'pi_cuda_gpu', 'pi_cuda_shared_memory', 'pi_cuda_2_level_reduction', 'pi_cuda_gpu_tableau']

# Lecture du fichier CSV - ORDRE CORRIGÉ
df = pd.read_csv('./Part_1/stats.csv', header=None, names=['version','num_steps','thredsPerBlock','runtime'], dtype={
                     'version': str,
                     'num_steps' : int,
                     'thredsPerBlock': int,
                     'runtime' : float
                 })

color_num_steps = {1000000 : "blue", 1000 : "red"}

# Nombre de subplots à créer : un pour chaque exécutable, cette fois disposés verticalement
fig, axes = plt.subplots(nrows=len(executables) + 1, ncols=1, figsize=(10, 5 * (len(executables) + 1)))

# Si il n'y a qu'un seul subplot (si executables ne contient qu'un seul élément)
if len(executables) == 1:
    axes = [axes]

# Style des lignes pour les différents exécutables
linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))]

# Tracer les courbes pour chaque exécutable dans son subplot respectif
for idx, executable in enumerate(executables):
    ax = axes[idx]
    
    for num_steps in sorted(df['num_steps'].unique()):
        df_plot = df[(df['num_steps'] == int(num_steps)) & (df['version'] == executable)]
        if df_plot.empty:
            continue
        
        # Grouper par thredsPerBlock et calculer la moyenne
        mean_stats = df_plot.groupby(['thredsPerBlock']).mean().reset_index()
        
        # Tracer la ligne moyenne
        ax.plot(mean_stats['thredsPerBlock'], mean_stats['runtime'], 
                linestyle='-', marker='o', color=color_num_steps[int(num_steps)],
                label=f'{num_steps} steps')
        
        # Tracer tous les points individuels
        ax.scatter(df_plot['thredsPerBlock'], df_plot['runtime'], 
                  color=color_num_steps[int(num_steps)], alpha=0.3, s=20)

    # Définir l'échelle des axes et les labels pour chaque subplot
    # ax.set_xscale('log', base=2)
    # ax.set_yscale('log')
    ax.set_xlabel('Threads per Block')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f'Runtime vs Threads per Block\n{executable}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

# Ajouter un graphe avec tous les exécutables pour num_steps=1000000
ax_all_executables = axes[-1]

for idx, executable in enumerate(executables):
    df_plot = df[(df['num_steps'] == 1000000) & (df['version'] == executable)]
    if df_plot.empty:
        continue
    
    # Grouper par thredsPerBlock et calculer la moyenne
    mean_stats = df_plot.groupby(['thredsPerBlock']).mean().reset_index()
    
    ls = linestyles[idx] if idx < len(linestyles) else 'solid'
    ax_all_executables.plot(mean_stats['thredsPerBlock'], mean_stats['runtime'], 
                           linestyle=ls, marker='o', label=executable)

# Définir l'échelle des axes et les labels pour le graphique global
# ax_all_executables.set_xscale('log', base=2)
# ax_all_executables.set_yscale('log')
ax_all_executables.set_xlabel('Threads per Block')
ax_all_executables.set_ylabel('Runtime (s)')
ax_all_executables.set_title('Comparison of All Executables (1000000 steps)')
ax_all_executables.grid(True, alpha=0.3)
ax_all_executables.legend(title='Executable', loc='best')

# Ajuster l'agencement des subplots
plt.tight_layout()

# Sauvegarder le plot en PNG
plt.savefig('./Part_1/performance_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved to ./Part_1/performance_analysis.png")

plt.show()