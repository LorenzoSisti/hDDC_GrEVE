"""
visualize_results.py
====================
Script per la visualizzazione della matrice GC e del grafo di rete.

Requisiti:
    pip install matplotlib networkx numpy
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ========================================================================
# CONFIGURAZIONE FILE
# ========================================================================
GC_MATRIX_PATH = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/weighted_graphs/holo_closed_GC_matrix.npy"
GRAPH_PKL_PATH = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/weighted_graphs/holo_closed_weighted.pkl"
OUTPUT_DIR = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/plots"


# ----------------------------------------------------------------------
# 1. Visualizzazione Heatmap Matrice GC
# ----------------------------------------------------------------------
def plot_gc_matrix(npy_path: str, save_path: str) -> None:
    """Carica la matrice .npy e genera la heatmap di correlazione."""
    matrix = np.load(npy_path)

    plt.figure(figsize=(8, 7))
    
    # Matrice plotted con origine in basso a sinistra (standard per mappe di contatto/correlazione)
    im = plt.imshow(matrix, cmap="viridis", origin="lower", vmin=0, vmax=1)
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Generalized Correlation (GC)", fontsize=11)

    plt.title("Matrice di Generalized Correlation", fontsize=13, pad=12)
    plt.xlabel("Indice Residuo", fontsize=11)
    plt.ylabel("Indice Residuo", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f" -> Heatmap matrice GC salvata in: {save_path}")
    plt.show()


# ----------------------------------------------------------------------
# 2. Visualizzazione 2D del Grafo NetworkX
# ----------------------------------------------------------------------
def plot_network_2d(
    pkl_path: str,
    save_path: str,
    gc_threshold: float = 0.3,
) -> None:
    """
    Carica il grafo .pkl e traccia una rappresentazione 2D.
    Applica una soglia di GC per visualizzare solo le connessioni significative
    ed evitare l'effetto "gomitolo" (hairball).
    """
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)

    # Crea un sottografo contenente solo gli archi con GC >= soglia
    subG = nx.Graph()
    for u, v, data in G.edges(data=True):
        if data.get("GC", 0.0) >= gc_threshold:
            subG.add_edge(u, v, **data)

    # Se ci sono nodi isolati dalla sfoltitura, manteniamo solo la componente connessa
    nodes_to_draw = list(subG.nodes())

    print(f"Grafo originale: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi.")
    print(f"Grafo filtrato (GC >= {gc_threshold}): {len(nodes_to_draw)} nodi, {subG.number_of_edges()} archi.")

    plt.figure(figsize=(10, 8))

    # Algoritmo di posizionamento 2D per il layout
    pos = nx.spring_layout(subG, k=0.15, seed=42)

    # Estrazione proprietà degli archi per la grafica
    gc_values = [subG[u][v].get("GC", 0.1) for u, v in subG.edges()]
    edge_widths = [val * 3.0 for val in gc_values]  # Spessore proporzionale a GC

    # Disegna nodi e archi
    nx.draw_networkx_nodes(
        subG, pos, node_size=25, node_color="#2b5c8f", alpha=0.9
    )
    nx.draw_networkx_edges(
        subG,
        pos,
        width=edge_widths,
        edge_color=gc_values,
        edge_cmap=plt.cm.Blues,
        alpha=0.6,
    )

    plt.title(
        f"Rete di Comunicazione Dinamica (Filtro GC ≥ {gc_threshold})",
        fontsize=13,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f" -> Grafo 2D salvato in: {save_path}")
    plt.show()


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Plot della Matrice
    plot_gc_matrix(
        npy_path=GC_MATRIX_PATH,
        save_path=out / "gc_heatmap.png",
    )

    # 2. Plot del Grafo 2D (mostra solo archi con GC >= 0.3)
    plot_network_2d(
        pkl_path=GRAPH_PKL_PATH,
        save_path=out / "network_2d.png",
        gc_threshold=0.3,
    )


if __name__ == "__main__":
    main()