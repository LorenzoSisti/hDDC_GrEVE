"""
time_resolved_activation_pathway.py
====================================

Step 3 della pipeline MD -> Teoria dei Grafi.
Analisi dinamica a finestre scorrevoli lungo la traiettoria di transizione (Simulazione C).

Protocollo a 3 Step:
  1. Discretizzazione temporale a finestre scorrevoli (sliding windows) della Simulazione C.
  2. Calcolo della Generalized Correlation (GC) via `dynetan` [1] e della Node Betweenness
     Centrality (BC_i(t)) per ciascuna finestra temporale per i residui chiave.
  3. Estrazione automatica di t_onset (istante in cui BC_i(t) supera il baseline di 3 sigma)
     e ordinamento della sequenza cronologica di attivazione allosterica.

Requisiti:
    pip install dynetan networkx mdanalysis pandas numpy
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import MDAnalysis as mda

import dynetan
from dynetan.proctraj import DNAproc


# ========================================================================
# CONFIGURAZIONE -- Modifica qui i percorsi e i parametri
# ========================================================================

# --- Traiettoria Simulazione C (Transizione Apo -> Holo) -----------------
TOPOLOGY_C = "sim_C.pdb"
TRAJECTORY_C = "sim_C.xtc"
SEG_IDS = ["A", "B"]  # Segment IDs presenti nella struttura

# --- File dei residui d'interesse (output dello Step 2) ------------------
KEY_RESIDUES_CSV = "./graphs/key_residues.csv"  # Deve contenere la colonna 'residue'

# --- Parametri Finestra Scorrevole (Sliding Window) ----------------------
WINDOW_SIZE_FRAMES = 1000    # Dimensione della finestra (es. 10 ns se 1 frame = 10 ps)
WINDOW_STRIDE_FRAMES = 100   # Passo di scorrimento (es. 1 ns tra finestre successive)
TIME_PER_FRAME_NS = 0.010    # Tempo reale tra i frame in nanosecondi (es. 0.010 ns = 10 ps)

# --- Parametri Dynetan / Grafi -------------------------------------------
CUTOFF_DIST = 4.5            # Distanza cutoff (Å) per i contatti
CONTACT_PERSISTENCE = 0.50   # Persistenza minima nella singola finestra
N_CORES = 8                  # Core CPU per il calcolo GC in parallelo
WEIGHT_EPSILON = 1e-6        # Floor numerico per evitare -log(0)

# --- Parametri Identificazione t_onset ----------------------------------
N_BASELINE_WINDOWS = 5       # Prime N finestre usate per calcolare media e std del baseline
SIGMA_THRESHOLD = 3.0        # Numero di deviazioni standard sopra il baseline per t_onset

# --- Output -------------------------------------------------------------
OUTPUT_DIR = "./time_series_output"


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def get_residue_label(g: nx.Graph, node) -> str:
    """Ricostruisce l'etichetta univoca del residuo (es. 'LYS_34_A')."""
    data = g.nodes[node]
    resname = data.get("resname", data.get("res_name", ""))
    resid = data.get("resid", node)
    chain = data.get("chain", data.get("chain_id", data.get("chainID", data.get("segid", ""))))

    parts = []
    if resname:
        parts.append(str(resname).upper())
    parts.append(str(resid))
    if chain:
        parts.append(str(chain).upper())

    label = "_".join(parts)
    return label if label else str(node)


def load_key_residues(csv_path: str) -> list[str]:
    """Carica la lista dei residui d'interesse dal file CSV prodotto dallo Step 2."""
    df = pd.read_csv(csv_path)
    if "residue" in df.columns:
        residues = df["residue"].dropna().astype(str).tolist()
    elif "res_i" in df.columns:
        residues = pd.concat([df["res_i"], df["res_j"]]).unique().tolist()
    else:
        raise KeyError(f"Impossibile trovare la colonna 'residue' o 'res_i' in {csv_path}")
    print(f" -> Caricati {len(residues)} residui chiave da monitorare.")
    return residues


# ----------------------------------------------------------------------
# 1. Elaborazione Singola Finestra Temporale con Dynetan
# ----------------------------------------------------------------------
def process_window_with_dynetan(
    topology: str,
    sub_trajectory: str,
    seg_ids: list[str],
    cutoff: float,
    persistence: float,
    ncores: int,
    epsilon: float,
) -> nx.Graph:
    """Esegue la pipeline dynetan su una sotto-traiettoria e restituisce il grafo pesato."""
    dnap = DNAproc()
    dnap.setNumWinds(1)
    dnap.setCutoffDist(cutoff)
    dnap.setContactPersistence(persistence)
    dnap.setSegIDs(seg_ids)

    dnap.loadSystem(topology, sub_trajectory)
    dnap.prepareNetwork()
    dnap.alignTraj(inMemory=True)
    dnap.findContacts(stride=1)
    dnap.filterContacts(notSameRes=True, notConsecutiveRes=False, removeIsolatedNodes=False)
    dnap.calcCor(ncores=ncores, verbose=0)
    dnap.calcGraphInfo()

    # Grafo NetworkX nativo e matrice GC
    G = dnap.nxGraphs[0]
    gc_matrix = dnap.corrMatAll[0]

    # Assegnazione dei pesi w_ij = -log(GC_ij)
    for u, v in G.edges():
        gc_val = float(gc_matrix[u, v])
        gc_clamped = max(gc_val, epsilon)
        G[u][v]["weight"] = float(-np.log(gc_clamped))

    return G


# ----------------------------------------------------------------------
# 2. Loop a Finestre Scorrate e Calcolo della Node Betweenness (BC_i(t))
# ----------------------------------------------------------------------
def run_sliding_window_analysis(
    topology: str,
    trajectory: str,
    key_residues: list[str],
    temp_dir: str,
) -> pd.DataFrame:
    """Scorre la traiettoria C in finestre temporali ed estrae la BC per i residui d'interesse."""
    u = mda.Universe(topology, trajectory)
    total_frames = len(u.trajectory)
    
    print(f"\n[Step 1 & 2] Analisi a finestre scorrevoli della Simulazione C:")
    print(f" -> Frame totali: {total_frames}")
    print(f" -> Dimensione finestra: {WINDOW_SIZE_FRAMES} frame ({WINDOW_SIZE_FRAMES * TIME_PER_FRAME_NS:.2f} ns)")
    print(f" -> Passo (stride): {WINDOW_STRIDE_FRAMES} frame ({WINDOW_STRIDE_FRAMES * TIME_PER_FRAME_NS:.2f} ns)")

    # Generazione degli intervalli delle finestre
    window_ranges = []
    start_f = 0
    while start_f + WINDOW_SIZE_FRAMES <= total_frames:
        end_f = start_f + WINDOW_SIZE_FRAMES
        window_ranges.append((start_f, end_f))
        start_f += WINDOW_STRIDE_FRAMES

    print(f" -> Numero totale di finestre temporali da calcolare: {len(window_ranges)}")

    records = []

    for win_idx, (s_frame, e_frame) in enumerate(window_ranges):
        mid_frame = (s_frame + e_frame) / 2.0
        time_ns = mid_frame * TIME_PER_FRAME_NS
        print(f"\n--- Finestra {win_idx + 1}/{len(window_ranges)} | Frame [{s_frame}:{e_frame}] | t_mid = {time_ns:.2f} ns ---")

        # Salva la sotto-traiettoria della finestra in un file temporaneo
        sub_xtc_path = os.path.join(temp_dir, f"window_{win_idx}.xtc")
        with mda.Writer(sub_xtc_path, u.atoms.n_atoms) as W:
            for ts in u.trajectory[s_frame:e_frame]:
                W.write(u.atoms)

        # Elabora la finestra con dynetan
        G = process_window_with_dynetan(
            topology=topology,
            sub_trajectory=sub_xtc_path,
            seg_ids=SEG_IDS,
            cutoff=CUTOFF_DIST,
            persistence=CONTACT_PERSISTENCE,
            ncores=N_CORES,
            epsilon=WEIGHT_EPSILON,
        )

        # Calcolo della Node Betweenness Centrality (BC_i(t))
        bc_dict_nodes = nx.betweenness_centrality(G, weight="weight")

        # Mappatura dei valori BC con l'etichetta del residuo
        bc_dict_labels = {get_residue_label(G, node): val for node, val in bc_dict_nodes.items()}

        # Estrazione dei soli residui d'interesse
        row = {"window_idx": win_idx, "frame_mid": mid_frame, "time_ns": time_ns}
        for res_label in key_residues:
            row[res_label] = bc_dict_labels.get(res_label, 0.0)

        records.append(row)

        # Rimuove il file temporaneo per liberare spazio
        if os.path.exists(sub_xtc_path):
            os.remove(sub_xtc_path)

    df_bc = pd.DataFrame(records)
    return df_bc


# ----------------------------------------------------------------------
# 3. Identificazione dei Tempi di Attivazione (t_onset)
# ----------------------------------------------------------------------
def compute_activation_sequence(
    df_bc: pd.DataFrame,
    key_residues: list[str],
    n_baseline_win: int = 5,
    sigma_thresh: float = 3.0,
) -> pd.DataFrame:
    """Calcola t_onset per ciascun residuo e ordina la sequenza di attivazione."""
    print(f"\n[Step 3] Determinazione dei tempi di attivazione (t_onset):")
    print(f" -> Baseline calcolato sulle prime {n_baseline_win} finestre.")
    print(f" -> Soglia di innesco: Mean_baseline + {sigma_thresh} * Std_baseline")

    activation_results = []

    for res in key_residues:
        bc_series = df_bc[res].values
        time_series = df_bc["time_ns"].values

        # Calcolo della baseline iniziale
        baseline_vals = bc_series[:n_baseline_win]
        mean_base = np.mean(baseline_vals)
        std_base = np.std(baseline_vals)
        threshold = mean_base + (sigma_thresh * std_base)

        # Trova il primo istante in cui la BC supera la soglia
        activated_indices = np.where(bc_series > threshold)[0]

        # Per evitare falsi positivi dovuti a fluttuazioni isolati, richiediamo che
        # l'attivazione rimanga sopra la soglia o che superi la media
        t_onset = np.nan
        onset_win = np.nan

        for idx in activated_indices:
            if idx >= n_baseline_win:  # Ignora scostamenti dentro la finestra baseline
                t_onset = time_series[idx]
                onset_win = idx
                break

        activation_results.append({
            "residue": res,
            "t_onset_ns": t_onset,
            "onset_window_idx": onset_win,
            "baseline_mean_BC": mean_base,
            "baseline_std_BC": std_base,
            "max_BC": np.max(bc_series),
        })

    df_act = pd.DataFrame(activation_results)

    # Ordina i residui in base al t_onset crescente (i primi a rispondere in cima)
    df_act = df_act.sort_values(by="t_onset_ns", ascending=True, na_position="last").reset_index(drop=True)
    return df_act


# ----------------------------------------------------------------------
# Main Execution Pipeline
# ----------------------------------------------------------------------
def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Caricamento residui d'interesse dal file dello Step 2
    key_residues = load_key_residues(KEY_RESIDUES_CSV)

    # 2. Esecuzione dell'analisi a finestre scorrevoli usando una cartella temporanea
    temp_dir = tempfile.mkdtemp(prefix="dynetan_windows_")
    try:
        df_bc_time_series = run_sliding_window_analysis(
            topology=TOPOLOGY_C,
            trajectory=TRAJECTORY_C,
            key_residues=key_residues,
            temp_dir=temp_dir,
        )
    finally:
        shutil.rmtree(temp_dir)  # Pulizia file temporanei

    # Salvataggio della matrice completa BC_i(t)
    bc_csv_path = out_dir / "bc_time_series.csv"
    df_bc_time_series.to_csv(bc_csv_path, index=False)
    print(f"\n -> Serie temporali BC_i(t) salvate in: {bc_csv_path}")

    # 3. Estrazione dei t_onset e della sequenza cronologica
    df_activation = compute_activation_sequence(
        df_bc=df_bc_time_series,
        key_residues=key_residues,
        n_baseline_win=N_BASELINE_WINDOWS,
        sigma_thresh=SIGMA_THRESHOLD,
    )

    # Salvataggio della sequenza di attivazione
    seq_csv_path = out_dir / "activation_sequence.csv"
    df_activation.to_csv(seq_csv_path, index=False)
    print(f" -> Sequenza di attivazione salvata in: {seq_csv_path}")

    # Stampa a terminale della sequenza cronologica ordinata
    print("\n==========================================================================")
    print("SEQUENZA CRONOLOGICA DI ATTIVAZIONE ALLOSTERICA (Simulazione C)")
    print("==========================================================================")
    
    activated_df = df_activation.dropna(subset=["t_onset_ns"])
    if not activated_df.empty:
        for rank, row in activated_df.iterrows():
            print(f"  {rank+1:2d}. Residuo {row['residue']:<15} | t_onset = {row['t_onset_ns']:.2f} ns (Max BC: {row['max_BC']:.4f})")
        
        sequence_str = " -> ".join(activated_df["residue"].tolist())
        print(f"\nCammino di propagazione temporale:\n  {sequence_str}")
    else:
        print("  Nessun residuo ha superato la soglia di attivazione specificata.")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()