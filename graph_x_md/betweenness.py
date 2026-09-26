"""
delta_betweenness_analysis.py
==============================

Variante di delta_eb_analysis.py che, invece della Edge Betweenness (EB),
calcola la Betweenness Centrality dei NODI (residui) nei due stati
conformazionali e la loro differenza (Delta_BC), per identificare i
residui piu' coinvolti nella riorganizzazione della rete allosterica.

    Delta_BC(residuo) = BC_closed(residuo) - BC_open(residuo)

Un Delta_BC positivo grande indica un residuo che diventa un hub di
comunicazione MOLTO piu' centrale nello stato chiuso rispetto all'aperto
(e viceversa per valori molto negativi).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ========================================================================
# CONFIGURAZIONE
# ========================================================================

APO_WEIGHTED_GRAPH = "/Users/lorenzosisti/GrEVE/invisible_data/weighted_graphs/apo_open_weighted.pkl"
HOLO_WEIGHTED_GRAPH = "/Users/lorenzosisti/GrEVE/invisible_data/weighted_graphs/holo_closed_weighted.pkl"

# BC calcolata sui cammini minimi pesati da -log(GC) (vedi script 1)
BC_WEIGHT_ATTRIBUTE: str | None = "weight"

# Soglia statistica (deviazioni standard) per marcare un residuo come significativo
Z_SCORE_THRESHOLD = 2.992

OUTPUT_DIR = "/Users/lorenzosisti/GrEVE/invisible_data/delta_bc_analysis"


# ----------------------------------------------------------------------
# Helper: etichetta univoca del residuo (dimeri/multimeri senza collisioni)
# ----------------------------------------------------------------------
def get_residue_label(g: nx.Graph, node) -> str:
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


# ----------------------------------------------------------------------
# 1. Caricamento grafo e calcolo BC indicizzata per etichetta residuo
# ----------------------------------------------------------------------
def load_weighted_graph(pickle_path: str) -> nx.Graph:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def betweenness_by_label(g: nx.Graph, weight_attribute: str | None) -> dict[str, float]:
    """
    Calcola la Betweenness Centrality dei nodi e la reindicizza usando
    l'etichetta completa del residuo (es. 'LYS_34_A').
    """
    bc_by_node = nx.betweenness_centrality(g, weight=weight_attribute, normalized=True)
    return {get_residue_label(g, node): value for node, value in bc_by_node.items()}


# ----------------------------------------------------------------------
# 2. Delta_BC sull'unione dei residui presenti nei due stati
# ----------------------------------------------------------------------
def compute_delta_bc_table(
    bc_open: dict[str, float], bc_closed: dict[str, float]
) -> pd.DataFrame:
    all_residues = set(bc_open.keys()) | set(bc_closed.keys())

    records = []
    for res in sorted(all_residues):
        bc_o = bc_open.get(res, 0.0)
        bc_c = bc_closed.get(res, 0.0)
        records.append(
            dict(residue=res, bc_open=bc_o, bc_closed=bc_c, delta_bc=bc_c - bc_o)
        )

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 3. Z-score di Delta_BC e flag dei residui significativi
# ----------------------------------------------------------------------
def add_z_score_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mean_delta = df["delta_bc"].mean()
    std_delta = df["delta_bc"].std()
    df["z_score"] = (df["delta_bc"] - mean_delta) / std_delta
    return df


def summarize_z_score_distribution(df: pd.DataFrame, output_dir: str) -> None:
    """Stampa un riepilogo della distribuzione di |z_score| e salva istogramma + CSV ordinato."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_z = df["z_score"].abs()

    percentiles = [50, 75, 90, 95, 97.5, 99, 99.5]
    print("    Distribuzione di |z_score| (Delta_BC) su tutti i residui:")
    print(f"        media Delta_BC = {df['delta_bc'].mean():.6f}, "
          f"deviazione standard = {df['delta_bc'].std():.6f}")
    for p in percentiles:
        value = np.percentile(abs_z, p)
        n_above = int((abs_z > value).sum())
        print(f"        percentile {p:>5.1f}%  ->  |z| = {value:.3f}  "
              f"({n_above} residui oltre questa soglia, {100 * n_above / len(abs_z):.1f}%)")

    # --- Istogramma ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["z_score"], bins=60, color="steelblue", edgecolor="white")
    ax.set_xlabel("z-score di Delta_BC")
    ax.set_ylabel("Numero di residui")
    ax.set_title("Distribuzione dello z-score di Delta_BC (tutti i residui)")

    candidate_thresholds = [2.0, 2.5, 3.0]
    colors = ["orange", "red", "darkred"]
    for threshold, color in zip(candidate_thresholds, colors):
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1, label=f"z = +/-{threshold}")
        ax.axvline(-threshold, color=color, linestyle="--", linewidth=1)
    ax.legend()

    histogram_path = out_dir / "z_score_distribution.png"
    fig.savefig(histogram_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> Istogramma salvato in: {histogram_path}")

    # --- Tabella completa ordinata ---
    sorted_path = out_dir / "z_score_sorted_all_residues.csv"
    df.assign(abs_z_score=abs_z).sort_values(
        "abs_z_score", ascending=False
    ).to_csv(sorted_path, index=False)
    print(f"    -> Tabella completa (ordinata per |z_score|) salvata in: {sorted_path}")


def flag_significant_residues(df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["significant"] = df["z_score"].abs() > z_threshold
    return df


# ----------------------------------------------------------------------
# 4. Salvataggio
# ----------------------------------------------------------------------
def save_results(df: pd.DataFrame, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_df = df.assign(abs_delta_bc=df["delta_bc"].abs()).sort_values(
        "abs_delta_bc", ascending=False
    ).drop(columns="abs_delta_bc")

    sorted_df.to_csv(out_dir / "delta_bc_all_residues.csv", index=False)
    sorted_df[sorted_df["significant"]].to_csv(
        out_dir / "delta_bc_significant_residues.csv", index=False
    )
    print(f"    -> Risultati salvati in: {out_dir}/")


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main() -> None:
    print("Caricamento dei grafi pesati...")
    g_open = load_weighted_graph(APO_WEIGHTED_GRAPH)
    g_closed = load_weighted_graph(HOLO_WEIGHTED_GRAPH)

    print("Calcolo della Betweenness Centrality (nodi) nei due stati...")
    bc_open = betweenness_by_label(g_open, BC_WEIGHT_ATTRIBUTE)
    bc_closed = betweenness_by_label(g_closed, BC_WEIGHT_ATTRIBUTE)

    print("Calcolo di Delta_BC sull'unione dei residui...")
    delta_bc_df = compute_delta_bc_table(bc_open, bc_closed)
    print(f"    -> {len(delta_bc_df)} residui totali (unione dei due stati).")

    delta_bc_df = add_z_score_column(delta_bc_df)
    summarize_z_score_distribution(delta_bc_df, OUTPUT_DIR)

    print(f"Identificazione dei residui sopra il rumore (|z| > {Z_SCORE_THRESHOLD})...")
    delta_bc_df = flag_significant_residues(delta_bc_df, Z_SCORE_THRESHOLD)
    n_significant = delta_bc_df["significant"].sum()
    print(f"    -> Con la soglia attuale (Z_SCORE_THRESHOLD={Z_SCORE_THRESHOLD}): "
          f"{n_significant} residui significativi su {len(delta_bc_df)}.")

    top10 = delta_bc_df.assign(abs_d=delta_bc_df["delta_bc"].abs()).sort_values(
        "abs_d", ascending=False
    ).head(10)
    print("\n    Top 10 residui per |Delta_BC|:")
    for _, row in top10.iterrows():
        print(f"        {row['residue']:>15s}  delta_bc={row['delta_bc']:+.5f}  z={row['z_score']:+.2f}")

    save_results(delta_bc_df, OUTPUT_DIR)

    print("\nCompletato.")


if __name__ == "__main__":
    main()
