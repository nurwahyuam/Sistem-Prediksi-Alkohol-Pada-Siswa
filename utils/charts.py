"""
charts.py
Generate grafik elbow dan cluster menggunakan matplotlib.
Simpan sebagai base64 PNG untuk ditampilkan di web.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure ke base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def plot_elbow(sse_list: list, optimal_k: int) -> str:
    """Buat grafik elbow method."""
    ks = [k for k, _ in sse_list]
    sses = [s for _, s in sse_list]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    ax.plot(ks, sses, 'o-', color='#38bdf8', linewidth=2.5,
            markersize=8, markerfacecolor='#f0abfc', markeredgecolor='#38bdf8',
            markeredgewidth=1.5, label='SSE')

    ax.axvline(x=optimal_k, color='#f59e0b', linestyle='--',
               linewidth=2, label=f'Optimal K={optimal_k}')
    ax.scatter([optimal_k], [sses[optimal_k - 1]], color='#f59e0b',
               s=150, zorder=5)

    ax.set_xlabel('Jumlah Cluster (K)', color='#94a3b8', fontsize=11)
    ax.set_ylabel('SSE (Inertia)', color='#94a3b8', fontsize=11)
    ax.set_title('Elbow Method — Penentuan K Optimal', color='#e2e8f0',
                 fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors='#64748b')
    ax.spines[:].set_color('#334155')
    ax.grid(True, color='#1e3a5f', linestyle='--', alpha=0.5)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    ax.set_xticks(ks)

    return fig_to_base64(fig)


def plot_clusters(df_with_cluster, feature_cols: list, k: int) -> str:
    """
    Visualisasi cluster menggunakan PCA 2D.
    """
    X = df_with_cluster[feature_cols].values
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    clusters = df_with_cluster['cluster'].values

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    colors = ['#38bdf8', '#f0abfc', '#fb923c', '#4ade80', '#facc15',
              '#f87171', '#a78bfa', '#34d399', '#e879f9', '#64748b']
    labels_cluster = [f'Cluster {i}' for i in range(k)]

    for c in range(k):
        mask = clusters == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=colors[c % len(colors)], alpha=0.75,
                   s=40, edgecolors='none', label=labels_cluster[c])

    ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                  color='#94a3b8', fontsize=10)
    ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                  color='#94a3b8', fontsize=10)
    ax.set_title(f'Visualisasi Cluster K-Means (K={k}) — PCA 2D',
                 color='#e2e8f0', fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors='#64748b')
    ax.spines[:].set_color('#334155')
    ax.grid(True, color='#1e3a5f', linestyle='--', alpha=0.4)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0',
              markerscale=1.5, fontsize=9)

    return fig_to_base64(fig)
