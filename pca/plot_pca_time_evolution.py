"""
plot_pca_time_evolution.py
==========================
Posizione script: hDDC_GrEVE/pca/
Input/Output:     hDDC_GrEVE/pca_results/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path


def load_xvg(filepath):
    """Carica un file XVG saltando le righe di intestazione (@ e #)."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("@") or not line.strip():
                continue
            data.append([float(x) for x in line.split()])
    return np.array(data)


def get_truncated_cmap(cmap_name, minval=0.35, maxval=1.0, n=256):
    """
    Crea una colormap troncata per evitare che i valori iniziali (0 ns)
    siano troppo chiari/bianchi e invisibili sul canvas.
    """
    cmap = plt.get_cmap(cmap_name)
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


# ==============================================================================
# GESTIONE PERCORSI E CONFIGURAZIONE
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "pca_results"
OUTPUT_PLOT = RESULTS_DIR / "fig_holo_open_vs_apo_open_time.png"

SIM_TIME_NS = 1000  # Durata totale di ciascuna simulazione in ns

# Colormap con gradienti troncati per le tre repliche
REPLICA_CMAPS = {
    "alessia": get_truncated_cmap("Blues", minval=0.35, maxval=1.0),
    "lorenzo": get_truncated_cmap("Oranges", minval=0.35, maxval=1.0),
    "sara": get_truncated_cmap("Greens", minval=0.35, maxval=1.0),
}

# Colori rappresentativi per la legenda
REPLICA_LEGEND_COLORS = {
    "alessia": "tab:blue",
    "lorenzo": "tab:orange",
    "sara": "tab:green",
}


def main():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"La cartella dei risultati non esiste: {RESULTS_DIR}")

    print(f"Cartella dello script:   {SCRIPT_DIR}")
    print(f"Cartella input/output:   {RESULTS_DIR}")

    # Creazione figura e axes espliciti
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    # --- 1. PLOT APO OPEN (SFONDO) ---
    apo_files = sorted(RESULTS_DIR.glob("proj2d_apo_open_apo_open_*.xvg"))
    if not apo_files:
        print(f"[WARNING] Nessun file 'proj2d_apo_open_apo_open_*.xvg' trovato in {RESULTS_DIR}")

    for i, fpath in enumerate(apo_files):
        data = load_xvg(fpath)
        pc1, pc2 = data[:, 0], data[:, 1]
        label = "apo_open" if i == 0 else None
        ax.scatter(
            pc1,
            pc2,
            color="lightgray",
            alpha=0.3,
            s=12,
            edgecolors="none",
            label=label,
            zorder=1,
        )

    # --- 2. PLOT HOLO OPEN CON EVOLUZIONE TEMPORALE ---
    holo_files = sorted(RESULTS_DIR.glob("proj2d_holo_open_holo_open_*.xvg"))
    if not holo_files:
        print(f"[WARNING] Nessun file 'proj2d_holo_open_holo_open_*.xvg' trovato in {RESULTS_DIR}")

    for fpath in holo_files:
        # Estrae il nome della replica (alessia, lorenzo, sara)
        replica_name = fpath.stem.split("_")[-1]
        data = load_xvg(fpath)
        pc1, pc2 = data[:, 0], data[:, 1]

        # Vettore tempo
        time_ns = np.linspace(0, SIM_TIME_NS, len(pc1))

        # Colormap dedicata
        cmap = REPLICA_CMAPS.get(
            replica_name, get_truncated_cmap("Purples", 0.35, 1.0)
        )

        ax.scatter(
            pc1,
            pc2,
            c=time_ns,
            cmap=cmap,
            s=18,
            alpha=0.75,
            edgecolors="none",
            zorder=2,
        )

    # --- 3. BARRA DEL TEMPO E LEGENDA ---
    # Barra del tempo neutra in scala di grigi
    sm = plt.cm.ScalarMappable(
        cmap=mcolors.LinearSegmentedColormap.from_list("gray_gradient", ["#d0d0d0", "#202020"]),
        norm=plt.Normalize(vmin=0, vmax=SIM_TIME_NS),
    )
    sm.set_array([])
    
    # Specificato ax=ax per evitare il ValueError
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Simulation Time (ns) [Light $\\rightarrow$ Dark]", fontsize=12)

    # Legenda personalizzata
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='apo_open (all)', markerfacecolor='lightgray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='holo_open (alessia)', markerfacecolor=REPLICA_LEGEND_COLORS['alessia'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='holo_open (lorenzo)', markerfacecolor=REPLICA_LEGEND_COLORS['lorenzo'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='holo_open (sara)', markerfacecolor=REPLICA_LEGEND_COLORS['sara'], markersize=8),
    ]

    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
    ax.set_xlabel("PC1 (nm)", fontsize=13)
    ax.set_ylabel("PC2 (nm)", fontsize=13)
    ax.set_title("holo_open vs apo_open — Time Evolution", fontsize=14, pad=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    # Salvataggio
    fig.savefig(OUTPUT_PLOT, dpi=300)
    plt.show()

    print(f"\n[OK] Grafico generato e salvato in:\n    {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()