import pandas as pd
import matplotlib.pyplot as plt

executables = ['matrix_sequential', 'matrix_cuda_gpu', 'matrix_cuda_shared_memory', 'matrix_cuda_2_level_reduction', 'matrix_cuda_shared_memory_optimized']


# Lecture du fichier CSV Part 2 (format: executable,ncore,n,m,runtime)
df = pd.read_csv('./Part_2/stats.csv', header=None,
                 names=['version','n','m','tpb','runtime'],
                 dtype={'version': str, 'n': int, 'm': int, 'tpb': int, 'runtime': float})

# nouvelle colonne s = n + m
df['s'] = df['n'] + df['m']

# mapping de couleurs pour chaque valeur de s
unique_s = sorted(df['s'].unique())
cmap = plt.get_cmap('tab10')
colors_s = {s: cmap(i % cmap.N) for i, s in enumerate(unique_s)}

# subplots : un par exécutable + un dernier récapitulatif
fig, axes = plt.subplots(nrows=len(executables) + 1, ncols=1, figsize=(8, 5 * (len(executables) + 1)))
if len(executables) == 1:
    axes = [axes]

linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))]

for idx, executable in enumerate(executables):
    ax = axes[idx]
    for s in unique_s:
        df_plot = df[(df['s'] == s) & (df['version'] == executable)]
        if df_plot.empty:
            continue
        mean_stats = df_plot.groupby(['s', 'version', 'tpb']).mean().reset_index()
        ls = linestyles[idx] if idx < len(linestyles) else 'solid'
        ax.plot(mean_stats['tpb'], mean_stats['runtime'], linestyle=ls, color=colors_s[s])
        ax.scatter(df_plot['tpb'], df_plot['runtime'], color=colors_s[s])

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('Threads per block')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f'Runtime vs Threads per block\n{executable}')

    # légende par valeur de s
    s_patches = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_s[s], markersize=8, label=f's={s} (n+m)')
        for s in unique_s
    ]
    version_handle = plt.Line2D([0], [0], linestyle=ls, color='black',
                                label=executable.replace('tp_openmp_part_2_vector_', ''))
    ax.legend(handles=[version_handle] + s_patches, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

# dernier subplot : tracer tous les exécutables pour une valeur de s choisie (ici la plus grande s)
ax_all_executables = axes[-1]
s_to_plot = unique_s[-1] if unique_s else None

if s_to_plot is not None:
    for executable in executables:
        df_plot = df[(df['s'] == s_to_plot) & (df['version'] == executable)]
        if df_plot.empty:
            continue
        mean_stats = df_plot.groupby(['s', 'version', 'tpb']).mean().reset_index()
        ls = linestyles[executables.index(executable)] if executables.index(executable) < len(linestyles) else 'solid'
        ax_all_executables.plot(mean_stats['tpb'], mean_stats['runtime'], linestyle=ls, label=executable)

    #ax_all_executables.set_xscale('log')
    #ax_all_executables.set_yscale('log')
    ax_all_executables.set_xlabel('Threads per block')
    ax_all_executables.set_ylabel('Runtime (s)')
    ax_all_executables.set_title(f'All Executables for s={s_to_plot} (n+m)')
    ax_all_executables.legend(title='Executable', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()

# Sauvegarder le plot en PNG
plt.savefig('./Part_2/performance_analysis.png', dpi=300, bbox_inches='tight')
print("Plot saved to ./Part_2/performance_analysis.png")

plt.subplots_adjust(right=0.75)
plt.show()