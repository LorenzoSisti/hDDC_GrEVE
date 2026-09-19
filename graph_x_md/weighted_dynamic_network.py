"""
dynetan_networkx_correlation_weighting.py
=========================================

Step 2 della pipeline MD -> Teoria dei Grafi.
Utilizza il pacchetto `dynetan` (Melo et al., 2020) lavorando interamente 
con grafi NetworkX.

Requisiti:
    pip install dynetan networkx mdanalysis
"""

from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import networkx as nx

# Importazione dei moduli di dynetan
import dynetan
from dynetan.proctraj import DNAproc


# ========================================================================
# CONFIGURAZIONE
# ========================================================================

# --- Sistema Apo / Open -------------------------------------------------
APO_TOPOLOGY = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/apo.pdb" 
APO_TRAJECTORY = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/apo_nowt.xtc"
APO_SEG_IDS = ["A", "B"]  # Segment ID(s) del sistema nell'universo MDAnalysis

# --- Sistema Holo / Closed ----------------------------------------------
HOLO_TOPOLOGY = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/holo.pdb"
HOLO_TRAJECTORY = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/holo_nowt.xtc"
HOLO_SEG_IDS = ["A", "B"]

# --- Parametri dell'analisi ---------------------------------------------
CORRELATION_STRIDE = 1      # Stride dei frame per il calcolo dei contatti/GC[cite: 1]
CUTOFF_DIST = 7           # Distanza di cutoff (Å) per la mappa di contatto[cite: 1]
CONTACT_PERSISTENCE = 0.75  # Persistenza minima del contatto (frazione di frame)[cite: 1]
N_CORES = 7                 # Numero di core per il calcolo parallelo della GC[cite: 1]
WEIGHT_EPSILON = 1e-6       # Floor numerico per evitare -log(0)

# --- Output -------------------------------------------------------------
OUTPUT_DIR = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/weighted_graphs"


# ----------------------------------------------------------------------
# 1. Pipeline di elaborazione con Dynetan
# ----------------------------------------------------------------------
def run_dynetan_pipeline(
    topology: str,
    trajectory: str,
    seg_ids: list[str],
    stride: int = 1,
    cutoff: float = 7.0,
    persistence: float = 0.75,
    ncores: int = 4,
) -> DNAproc:
    """
    Esegue il flusso di lavoro standard di dynetan tramite l'oggetto DNAproc:
    1. Inizializzazione parametri
    2. Caricamento topologia e traiettoria
    3. Allineamento
    4. Calcolo e filtraggio contatti
    5. Calcolo della Generalized Correlation
    """
    dnap = DNAproc()

    # Impostazione della configurazione
    dnap.setNumWinds(1)  # 1 singola finestra per l'intera traiettoria
    dnap.setCutoffDist(cutoff)
    dnap.setContactPersistence(persistence)
    dnap.setSegIDs(seg_ids)

    # Caricamento del sistema
    print(f"Caricamento del sistema: {topology}, {trajectory}")
    dnap.loadSystem(topology, trajectory)

    # Preparazione dei nodi della rete (1 nodo per residuo proteico / C-alpha)
    dnap.prepareNetwork()

    # Allineamento della traiettoria (in memoria se possibile)
    print("Allineamento della traiettoria...")
    dnap.alignTraj(inMemory=True)

    # Individuazione dei contatti lungo la traiettoria
    print(f"Calcolo dei contatti (cutoff={cutoff}Å, stride={stride})...")
    dnap.findContacts(stride=stride)

    # Filtraggio dei contatti intra-residuo e isolati
    dnap.filterContacts(
        notSameRes=True, notConsecutiveRes=False, removeIsolatedNodes=False
    )

    # Calcolo della Generalized Correlation (GC) in parallelo
    print(f"Calcolo della Generalized Correlation su {ncores} cores...")
    dnap.calcCor(ncores=ncores, verbose=1)

    return dnap


# ----------------------------------------------------------------------
# 2. Arricchimento del Grafo NetworkX e Salvataggio
# ----------------------------------------------------------------------
def process_and_save_networkx(
    dnap: DNAproc, output_dir: str, label: str, epsilon: float = 1e-6
) -> None:
    """
    Estrae il grafo NetworkX generato da dynetan, assegna gli attributi
    'GC' e 'weight' a ciascun arco, e lo salva sia in formato .graphml che .pkl.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Salva la matrice GC completa in formato numpy (.npy)
    gc_matrix = dnap.corrMatAll[0]  # Matrice N x N della finestra 0
    npy_path = out_dir / f"{label}_GC_matrix.npy"
    np.save(npy_path, gc_matrix)
    print(f" -> Matrice GC salvata in: {npy_path}")

    # 2. Inizializza la struttura del grafo in dynetan
    dnap.calcGraphInfo()
    G = dnap.nxGraphs[0]  # Grafo networkx.Graph nativo generato da dynetan

    # 3. Calcola e assegna gli attributi 'GC' e 'weight' = -log(GC) direttamente agli archi
    for u, v in G.edges():
        gc_val = float(gc_matrix[u, v])
        gc_clamped = max(gc_val, epsilon)
        weight_val = float(-np.log(gc_clamped))

        G[u][v]["GC"] = gc_val
        G[u][v]["weight"] = weight_val

    # 4. Salvataggio in formato Pickle (.pkl)
    pickle_path = out_dir / f"{label}_weighted.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(G, f)
    print(f" -> Grafo NetworkX salvato (.pkl): {pickle_path}")

    # 5. Salvataggio in formato GraphML (.graphml)
    graphml_path = out_dir / f"{label}_weighted.graphml"
    G_graphml = G.copy()

    for _, data in G_graphml.nodes(data=True):
        for k, val in list(data.items()):
            if isinstance(val, (set, list, tuple, dict)):
                data[k] = str(val)

    for _, _, data in G_graphml.edges(data=True):
        for k, val in list(data.items()):
            if isinstance(val, (set, list, tuple, dict)):
                data[k] = str(val)

    nx.write_graphml(G_graphml, str(graphml_path))
    print(f" -> Grafo NetworkX salvato (.graphml): {graphml_path}")


# ----------------------------------------------------------------------
# 3. Esecuzione del Flusso di Lavoro
# ----------------------------------------------------------------------
def process_system(
    topology: str,
    trajectory: str,
    seg_ids: list[str],
    label: str,
) -> None:
    print(f"\n=================== Sistema: {label} ===================")

    # Esegue l'analisi con dynetan
    dnap = run_dynetan_pipeline(
        topology=topology,
        trajectory=trajectory,
        seg_ids=seg_ids,
        stride=CORRELATION_STRIDE,
        cutoff=CUTOFF_DIST,
        persistence=CONTACT_PERSISTENCE,
        ncores=N_CORES,
    )

    # Elabora e salva il grafo in NetworkX
    process_and_save_networkx(
        dnap=dnap,
        output_dir=OUTPUT_DIR,
        label=label,
        epsilon=WEIGHT_EPSILON,
    )


def main() -> None:
    systems = [
        dict(
            topology=APO_TOPOLOGY,
            trajectory=APO_TRAJECTORY,
            seg_ids=APO_SEG_IDS,
            label="apo_open",
        ),
        dict(
            topology=HOLO_TOPOLOGY,
            trajectory=HOLO_TRAJECTORY,
            seg_ids=HOLO_SEG_IDS,
            label="holo_closed",
        ),
    ]

    for sys_info in systems:
        process_system(**sys_info)

    print("\nCompletato. Flusso di lavoro puramente NetworkX concluso con successo.")


if __name__ == "__main__":
    main()