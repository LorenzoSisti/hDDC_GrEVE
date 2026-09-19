"""
delta_eb_analysis.py
======================

Step 3 della pipeline MD -> Teoria dei Grafi (roadmap, Fase 1 "Rete di
comunicazione allosterica differenziale (Delta_EB)" e Fase 3.3).

Confronta i grafi pesati di due stati conformazionali (Apo/Open e
Holo/Closed) e:

    1. Calcola la Edge Betweenness Centrality (EB) separatamente nei due
       stati.
    2. Identifica i residui con etichetta completa (es. LYS_34_A) per 
       gestire correttamente dimeri/multimeri senza collisioni di resid.
    3. Calcola Delta_EB = EB_closed - EB_open per ogni contatto (edge).
    4. Identifica i contatti significativi sopra la soglia Z-score.
    5. Ricostruisce i pathway di transizione (componenti connesse).
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

APO_WEIGHTED_GRAPH = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/weighted_graphs/apo_open_weighted.pkl"
HOLO_WEIGHTED_GRAPH = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/weighted_graphs/holo_closed_weighted.pkl"

# EB calcolata sui cammini minimi PESATI dalla correlazione (w_ij = -log(GC_ij))
EB_WEIGHT_ATTRIBUTE: str | None = "weight"

# Soglia statistica (in deviazioni standard) - Z = 3.793 corrisponde al top 1% (99° percentile)
Z_SCORE_THRESHOLD = 3.889 

OUTPUT_DIR = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/delta_eb_analysis"


# ----------------------------------------------------------------------
# Helper: Generazione etichetta univoca per dimeri/multimeri
# ----------------------------------------------------------------------
def get_residue_label(g: nx.Graph, node) -> str:
    """
    Costruisce un'etichetta univoca e parlante per il residuo (es. 'LYS_34_A').
    Ispeziona gli attributi del nodo nel grafo NetworkX (resname, resid, chain/chain_id/segid).
    """
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
# 1. Caricamento dei grafi e calcolo EB indicizzata per etichetta residuo
# ----------------------------------------------------------------------
def load_weighted_graph(pickle_path: str) -> nx.Graph:
    """Carica il grafo NetworkX pesato salvato dallo script precedente."""
    with open(pickle_path, "rb") as f:
        g = pickle.load(f)

    # DIAGNOSTICA: Stampa gli attributi del primo nodo per vedere le chiavi reali
    first_node = next(iter(g.nodes()))
    print(f"\n[DEBUG] Attributi del nodo '{first_node}' in {pickle_path}:")
    print(g.nodes[first_node])
    print("-" * 50)

    return g


def edge_betweenness_by_label(
    g: nx.Graph, weight_attribute: str | None
) -> dict[tuple[str, str], float]:
    """
    Calcola la Edge Betweenness Centrality sui nodi del grafo e la reindicizza
    utilizzando l'etichetta completa del residuo (es. 'LYS_34_A'), prevenendo
    collisioni di indici tra le catene del dimero.
    """
    eb_by_node_index = nx.edge_betweenness_centrality(g, weight=weight_attribute)

    eb_by_label: dict[tuple[str, str], float] = {}
    for (node_i, node_j), value in eb_by_node_index.items():
        label_i = get_residue_label(g, node_i)
        label_j = get_residue_label(g, node_j)
        key = (label_i, label_j) if label_i < label_j else (label_j, label_i)
        eb_by_label[key] = value

    return eb_by_label


# ----------------------------------------------------------------------
# 2. Calcolo di Delta_EB sull'unione dei contatti dei due stati
# ----------------------------------------------------------------------
def compute_delta_eb_table(
    eb_open: dict[tuple[str, str], float],
    eb_closed: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """
    Costruisce una tabella con una riga per ogni contatto presente in
    ALMENO UNO dei due stati.
    """
    all_edges = set(eb_open.keys()) | set(eb_closed.keys())

    records = []
    for res_i, res_j in sorted(all_edges):
        eb_o = eb_open.get((res_i, res_j), 0.0)
        eb_c = eb_closed.get((res_i, res_j), 0.0)
        records.append(
            dict(res_i=res_i, res_j=res_j, eb_open=eb_o, eb_closed=eb_c, delta_eb=eb_c - eb_o)
        )

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 3. Distribuzione dello z-score di Delta_EB: calcolo, diagnostica, soglia
# ----------------------------------------------------------------------
def add_z_score_column(delta_eb_df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge la colonna 'z_score' a delta_eb_df."""
    mean_delta = delta_eb_df["delta_eb"].mean()
    std_delta = delta_eb_df["delta_eb"].std()

    delta_eb_df = delta_eb_df.copy()
    delta_eb_df["z_score"] = (delta_eb_df["delta_eb"] - mean_delta) / std_delta
    return delta_eb_df


def summarize_z_score_distribution(delta_eb_df: pd.DataFrame, output_dir: str) -> None:
    """Stampa un riepilogo della distribuzione di |z_score| e salva grafici e CSV."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_z = delta_eb_df["z_score"].abs()

    percentiles = [50, 75, 90, 95, 97.5, 99, 99.5]
    print("    Distribuzione di |z_score| (Delta_EB) su tutti gli edge:")
    print(f"        media Delta_EB = {delta_eb_df['delta_eb'].mean():.5f}, "
          f"deviazione standard = {delta_eb_df['delta_eb'].std():.5f}")
    for p in percentiles:
        value = np.percentile(abs_z, p)
        n_above = int((abs_z > value).sum())
        print(f"        percentile {p:>5.1f}%  ->  |z| = {value:.3f}  "
              f"({n_above} edge oltre questa soglia, {100 * n_above / len(abs_z):.1f}%)")

    # --- Istogramma ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_eb_df["z_score"], bins=60, color="steelblue", edgecolor="white")
    ax.set_xlabel("z-score di Delta_EB")
    ax.set_ylabel("Numero di edge")
    ax.set_title("Distribuzione dello z-score di Delta_EB (tutti gli edge)")

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
    sorted_path = out_dir / "z_score_sorted_all_edges.csv"
    delta_eb_df.assign(abs_z_score=abs_z).sort_values(
        "abs_z_score", ascending=False
    ).to_csv(sorted_path, index=False)
    print(f"    -> Tabella completa (ordinata per |z_score|) salvata in: {sorted_path}")


def flag_significant_edges(delta_eb_df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """Flagga gli archi con |z_score| > z_threshold."""
    delta_eb_df = delta_eb_df.copy()
    delta_eb_df["significant"] = delta_eb_df["z_score"].abs() > z_threshold
    return delta_eb_df


# ----------------------------------------------------------------------
# 4. Residui chiave e pathway della transizione (componenti connesse)
# ----------------------------------------------------------------------
def find_key_residues_and_pathways(
    delta_eb_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[nx.Graph]]:
    """Costruisce il sottografo degli edge significativi ed estrae i pathway."""
    significant_edges = delta_eb_df[delta_eb_df["significant"]]

    significant_subgraph = nx.Graph()
    for _, row in significant_edges.iterrows():
        significant_subgraph.add_edge(
            str(row["res_i"]), str(row["res_j"]), delta_eb=row["delta_eb"]
        )

    importance: dict[str, float] = {}
    edge_count: dict[str, int] = {}
    for res_i, res_j, data in significant_subgraph.edges(data=True):
        contribution = abs(data["delta_eb"])
        importance[res_i] = importance.get(res_i, 0.0) + contribution
        importance[res_j] = importance.get(res_j, 0.0) + contribution
        edge_count[res_i] = edge_count.get(res_i, 0) + 1
        edge_count[res_j] = edge_count.get(res_j, 0) + 1

    key_residues_df = pd.DataFrame(
        [
            dict(residue=res, importance_score=importance[res], n_significant_edges=edge_count[res])
            for res in importance
        ]
    ).sort_values("importance_score", ascending=False).reset_index(drop=True)

    components = sorted(
        nx.connected_components(significant_subgraph), key=len, reverse=True
    )
    pathway_subgraphs = [significant_subgraph.subgraph(component).copy() for component in components]

    return key_residues_df, pathway_subgraphs


# ----------------------------------------------------------------------
# 5. Salvataggio dei risultati
# ----------------------------------------------------------------------
def save_results(
    delta_eb_df: pd.DataFrame,
    key_residues_df: pd.DataFrame,
    pathway_subgraphs: list[nx.Graph],
    output_dir: str,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_eb_df.to_csv(out_dir / "delta_eb_all_edges.csv", index=False)
    delta_eb_df[delta_eb_df["significant"]].to_csv(
        out_dir / "delta_eb_significant_edges.csv", index=False
    )
    key_residues_df.to_csv(out_dir / "key_residues.csv", index=False)

    with open(out_dir / "transition_pathways.txt", "w") as f:
        f.write(f"Trovate {len(pathway_subgraphs)} componenti connesse "
                f"(pathway candidati) fra gli edge significativi.\n\n")
        for idx, subgraph in enumerate(pathway_subgraphs):
            residues = sorted(subgraph.nodes())
            f.write(f"--- Pathway {idx} ({len(residues)} residui, "
                     f"{subgraph.number_of_edges()} contatti) ---\n")
            f.write(f"Residui: {residues}\n")
            f.write("Contatti (res_i, res_j, delta_eb):\n")
            for res_i, res_j, data in subgraph.edges(data=True):
                f.write(f"    {res_i} -- {res_j} : {data['delta_eb']:.4f}\n")
            f.write("\n")

    print(f"    -> Risultati salvati in: {out_dir}/")


# ----------------------------------------------------------------------
# 6. Funzione principale
# ----------------------------------------------------------------------
def main() -> None:
    print("Caricamento dei grafi pesati...")
    g_open = load_weighted_graph(APO_WEIGHTED_GRAPH)
    g_closed = load_weighted_graph(HOLO_WEIGHTED_GRAPH)

    print("Calcolo della Edge Betweenness nei due stati con etichette complete...")
    eb_open = edge_betweenness_by_label(g_open, EB_WEIGHT_ATTRIBUTE)
    eb_closed = edge_betweenness_by_label(g_closed, EB_WEIGHT_ATTRIBUTE)

    print("Calcolo di Delta_EB sull'unione dei contatti...")
    delta_eb_df = compute_delta_eb_table(eb_open, eb_closed)
    print(f"    -> {len(delta_eb_df)} contatti totali (unione dei due stati).")

    print(f"Identificazione degli edge sopra il rumore (|z| > {Z_SCORE_THRESHOLD})...")
    delta_eb_df = add_z_score_column(delta_eb_df)
    summarize_z_score_distribution(delta_eb_df, OUTPUT_DIR)
    delta_eb_df = flag_significant_edges(delta_eb_df, Z_SCORE_THRESHOLD)
    n_significant = delta_eb_df["significant"].sum()
    print(f"    -> Con la soglia attuale (Z_SCORE_THRESHOLD={Z_SCORE_THRESHOLD}): "
          f"{n_significant} edge significativi su {len(delta_eb_df)}.")

    print("Identificazione dei residui chiave e dei pathway di transizione...")
    key_residues_df, pathway_subgraphs = find_key_residues_and_pathways(delta_eb_df)
    print(f"    -> {len(key_residues_df)} residui chiave.")
    print(f"    -> {len(pathway_subgraphs)} pathway (componenti connesse) candidati.")
    if pathway_subgraphs:
        largest = pathway_subgraphs[0]
        print(f"    -> Il pathway piu' esteso coinvolge {largest.number_of_nodes()} "
              f"residui: {sorted(largest.nodes())}")

    save_results(delta_eb_df, key_residues_df, pathway_subgraphs, OUTPUT_DIR)

    print("\nCompletato.")


if __name__ == "__main__":
    main()