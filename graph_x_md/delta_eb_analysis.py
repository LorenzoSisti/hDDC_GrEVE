"""
delta_eb_analysis.py
======================

Step 3 della pipeline MD -> Teoria dei Grafi (roadmap, Fase 1 "Rete di
comunicazione allosterica differenziale (Delta_EB)" e Fase 3.3).

Confronta i grafi pesati di due stati conformazionali (Apo/Open e
Holo/Closed) e:

    1. Calcola la Edge Betweenness Centrality (EB) separatamente nei due
       stati (NON pesata, weight=None -- stessa convenzione di
       dynetan.calcBetween(): conta solo il NUMERO di cammini minimi
       topologici che attraversano ciascun edge).
    2. Calcola Delta_EB = EB_closed - EB_open per ogni contatto (edge)
       presente in almeno uno dei due stati (un contatto assente in uno
       stato vale EB=0 li').
    3. Identifica i contatti "sopra il rumore": qui il "rumore" e' la
       distribuzione stessa di Delta_EB su tutta la rete (la maggior
       parte dei contatti e' conservata, quindi Delta_EB ~ 0). Prima di
       applicare una soglia, la distribuzione dello z-score viene
       riassunta (percentili + istogramma salvato su disco) cosi' la
       soglia si sceglie guardando i dati reali, non un valore standard
       preimpostato. Un edge e' considerato significativo se il suo
       z-score supera la soglia scelta (Z_SCORE_THRESHOLD).
    4. Identifica i "residui chiave per la transizione": i residui
       toccati da almeno un edge significativo.
    5. Ricostruisce il/i "pathway della transizione": le componenti
       connesse del sottografo formato SOLO dagli edge significativi.
       Ogni componente e' una catena contigua di residui il cui contatto
       cambia in modo coordinato fra i due stati -- il concetto di
       "contiguous dynamic pathway" descritto nella review allegata
       (Patel et al. 2024, Fig. 4b).

NOTA IMPORTANTE (a differenza del metodo SNR visto in precedenza): questo
approccio non richiede di definire a priori residui "sorgente"/
"destinazione". Emerge direttamente dal confronto dei due stati.

Il codice segue lo stile della documentazione ufficiale di:
    * NetworkX -> https://networkx.org/documentation/stable/
      (in particolare nx.edge_betweenness_centrality,
      nx.connected_components)

Riferimenti:
    * Nierzwicki et al., eLife 2021 (Delta_EB fra sistema mutato e WT --
      stessa logica qui applicata a due stati conformazionali)
    * Patel, Sinha & Palermo, Q. Rev. Biophys. 2024, Fig. 4-5
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ========================================================================
# CONFIGURAZIONE -- modifica qui i path e i parametri, poi lancia
# semplicemente: python delta_eb_analysis.py
# ========================================================================

APO_WEIGHTED_GRAPH = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/weighted_graphs/apo_open_weighted.pkl"
HOLO_WEIGHTED_GRAPH = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/weighted_graphs/holo_closed_weighted.pkl"

# EB calcolata sui cammini minimi TOPOLOGICI (weight=None), come in
# dynetan.calcBetween(). Se preferisci usare i cammini minimi PESATI
# dalla correlazione (w_ij = -log(GC_ij)), imposta EB_WEIGHT_ATTRIBUTE
# = "weight" invece di None.

#EB_WEIGHT_ATTRIBUTE: str | None = None
EB_WEIGHT_ATTRIBUTE: str | None = "weight"

# Soglia statistica (in deviazioni standard) oltre la quale un edge e'
# considerato "sopra il rumore" di Delta_EB. Valori tipici: 2.0-3.0.
Z_SCORE_THRESHOLD = 3.793 #top 1%  # 3.793 = 99-esimo percentile della distribuzione di |z_score| su tutti gli edge

OUTPUT_DIR = "/Users/lorenzosisti/hDDC_GrEVE/invisible_data/delta_eb_analysis"


# ----------------------------------------------------------------------
# 1. Caricamento dei grafi e calcolo della EB indicizzata per resid
# ----------------------------------------------------------------------
def load_weighted_graph(pickle_path: str) -> nx.Graph:
    """Carica il grafo NetworkX pesato salvato dallo script precedente."""
    with open(pickle_path, "rb") as f:
        g = pickle.load(f)

    if "resid" not in next(iter(g.nodes(data=True)))[1]:
        raise ValueError(
            f"Il grafo '{pickle_path}' non ha l'attributo di nodo 'resid'. "
            "Aggiungi la patch (nx.set_node_attributes con dnap.nodesAtmSel) "
            "al tuo script di pesatura e rigenera il file."
        )
    return g


def edge_betweenness_by_resid(
    g: nx.Graph, weight_attribute: str | None
) -> dict[tuple[int, int], float]:
    """
    Calcola la Edge Betweenness Centrality e la reindicizza per RESID
    (invece che per indice di nodo interno), cosi' da poter confrontare
    direttamente grafi con ordinamento dei nodi diverso.

    Parameters
    ----------
    g : nx.Graph
        Deve avere l'attributo di nodo 'resid'.
    weight_attribute : str | None
        Nome dell'attributo di edge da usare come peso per i cammini
        minimi (None = non pesato, come in dynetan).

    Returns
    -------
    dict[(int, int), float]
        Chiave: tupla (resid_i, resid_j) ordinata (resid_i < resid_j).
        Valore: Edge Betweenness Centrality.
    """
    eb_by_node_index = nx.edge_betweenness_centrality(g, weight=weight_attribute)

    eb_by_resid: dict[tuple[int, int], float] = {}
    for (node_i, node_j), value in eb_by_node_index.items():
        resid_i = g.nodes[node_i]["resid"]
        resid_j = g.nodes[node_j]["resid"]
        key = (resid_i, resid_j) if resid_i < resid_j else (resid_j, resid_i)
        eb_by_resid[key] = value

    return eb_by_resid


# ----------------------------------------------------------------------
# 2. Calcolo di Delta_EB sull'unione dei contatti dei due stati
# ----------------------------------------------------------------------
def compute_delta_eb_table(
    eb_open: dict[tuple[int, int], float],
    eb_closed: dict[tuple[int, int], float],
) -> pd.DataFrame:
    """
    Costruisce una tabella con una riga per ogni contatto presente in
    ALMENO UNO dei due stati (un contatto assente in uno stato ha EB=0
    li', perche' quel canale di comunicazione semplicemente non esiste).

    Returns
    -------
    pd.DataFrame
        Colonne: resid_i, resid_j, eb_open, eb_closed, delta_eb.
    """
    all_edges = set(eb_open.keys()) | set(eb_closed.keys())

    records = []
    for resid_i, resid_j in sorted(all_edges):
        eb_o = eb_open.get((resid_i, resid_j), 0.0)
        eb_c = eb_closed.get((resid_i, resid_j), 0.0)
        records.append(
            dict(resid_i=resid_i, resid_j=resid_j, eb_open=eb_o, eb_closed=eb_c, delta_eb=eb_c - eb_o)
        )

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# 3. Distribuzione dello z-score di Delta_EB: calcolo, diagnostica, soglia
# ----------------------------------------------------------------------
def add_z_score_column(delta_eb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge alla tabella la colonna 'z_score': lo z-score di delta_eb
    rispetto alla distribuzione su TUTTI gli edge (che qui rappresenta il
    "rumore" di fondo, dato che la maggior parte dei contatti e'
    conservata fra i due stati e ha Delta_EB vicino a zero).

    Separata da flag_significant_edges() apposta, cosi' puoi ispezionare
    la distribuzione (vedi summarize_z_score_distribution) PRIMA di
    decidere la soglia da usare.
    """
    mean_delta = delta_eb_df["delta_eb"].mean()
    std_delta = delta_eb_df["delta_eb"].std()

    delta_eb_df = delta_eb_df.copy()
    delta_eb_df["z_score"] = (delta_eb_df["delta_eb"] - mean_delta) / std_delta
    return delta_eb_df


def summarize_z_score_distribution(delta_eb_df: pd.DataFrame, output_dir: str) -> None:
    """
    Stampa un riepilogo della distribuzione di |z_score| e salva:
        - un istogramma (.png) della distribuzione di Delta_EB, con
          linee verticali alle soglie candidate (2.0, 2.5, 3.0 std, e il
          95-esimo/99-esimo percentile di |z_score|);
        - la tabella completa ordinata per |z_score| decrescente (.csv),
          utile per scorrere manualmente "dove si stacca la coda" della
          distribuzione.

    Non decide nulla al posto tuo: e' pensata per essere letta prima di
    impostare Z_SCORE_THRESHOLD nella sezione CONFIGURAZIONE.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_z = delta_eb_df["z_score"].abs()

    percentiles = [50, 75, 90, 95, 97.5, 99, 99.5]
    print("    Distribuzione di |z_score| (Delta_EB) su tutti gli edge:")
    print(f"        media Delta_EB = {delta_eb_df['delta_eb'].mean():.5f}, "
          f"deviazione standard = {delta_eb_df['delta_eb'].std():.5f}")
    for p in percentiles:
        value = np.percentile(abs_z, p)
        # Quanti edge (e che frazione) verrebbero marcati "significativi"
        # se si usasse proprio questo percentile come soglia.
        n_above = int((abs_z > value).sum())
        print(f"        percentile {p:>5.1f}%  ->  |z| = {value:.3f}  "
              f"({n_above} edge oltre questa soglia, {100 * n_above / len(abs_z):.1f}%)")

    # --- Istogramma con soglie candidate marcate -----------------------
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

    # --- Tabella ordinata per |z_score|, per ispezione manuale ---------
    sorted_path = out_dir / "z_score_sorted_all_edges.csv"
    delta_eb_df.assign(abs_z_score=abs_z).sort_values(
        "abs_z_score", ascending=False
    ).to_csv(sorted_path, index=False)
    print(f"    -> Tabella completa (ordinata per |z_score|) salvata in: {sorted_path}")


def flag_significant_edges(delta_eb_df: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    """
    Aggiunge la colonna booleana 'significant' alla tabella (che deve gia'
    avere la colonna 'z_score', vedi add_z_score_column). Un edge e'
    'significant' se |z_score| > z_threshold: il suo Delta_EB si scosta
    dalla media molto piu' di quanto ci si aspetterebbe dalla normale
    variabilita' del network fra due stati.
    """
    delta_eb_df = delta_eb_df.copy()
    delta_eb_df["significant"] = delta_eb_df["z_score"].abs() > z_threshold
    return delta_eb_df


# ----------------------------------------------------------------------
# 4. Residui chiave e pathway della transizione (componenti connesse)
# ----------------------------------------------------------------------
def find_key_residues_and_pathways(
    delta_eb_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[nx.Graph]]:
    """
    A partire dalla tabella con la colonna 'significant', costruisce il
    sottografo fatto SOLO dagli edge significativi e ne estrae:
        - i residui chiave (nodi di questo sottografo), con un punteggio
          di importanza pari alla somma di |Delta_EB| degli edge
          significativi a cui partecipano;
        - le componenti connesse (i "pathway" candidati), come lista di
          sotto-grafi NetworkX, ordinate dalla piu' grande alla piu'
          piccola.

    Returns
    -------
    key_residues_df : pd.DataFrame
        Colonne: resid, importance_score, n_significant_edges. Ordinato
        per importance_score decrescente.
    pathway_subgraphs : list[nx.Graph]
        Un sottografo per componente connessa, con l'attributo di edge
        'delta_eb' preservato. Ordinati dal piu' grande al piu' piccolo.
    """
    significant_edges = delta_eb_df[delta_eb_df["significant"]]

    significant_subgraph = nx.Graph()
    for _, row in significant_edges.iterrows():
        significant_subgraph.add_edge(
            int(row["resid_i"]), int(row["resid_j"]), delta_eb=row["delta_eb"]
        )

    # --- Punteggio di importanza per residuo ------------------------------
    importance: dict[int, float] = {}
    edge_count: dict[int, int] = {}
    for resid_i, resid_j, data in significant_subgraph.edges(data=True):
        contribution = abs(data["delta_eb"])
        importance[resid_i] = importance.get(resid_i, 0.0) + contribution
        importance[resid_j] = importance.get(resid_j, 0.0) + contribution
        edge_count[resid_i] = edge_count.get(resid_i, 0) + 1
        edge_count[resid_j] = edge_count.get(resid_j, 0) + 1

    key_residues_df = pd.DataFrame(
        [
            dict(resid=resid, importance_score=importance[resid], n_significant_edges=edge_count[resid])
            for resid in importance
        ]
    ).sort_values("importance_score", ascending=False).reset_index(drop=True)

    # --- Componenti connesse = pathway candidati ---------------------------
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

    # Un file di testo leggibile con l'elenco dei pathway (componenti
    # connesse), utile per un controllo rapido senza dover riaprire i CSV.
    with open(out_dir / "transition_pathways.txt", "w") as f:
        f.write(f"Trovate {len(pathway_subgraphs)} componenti connesse "
                f"(pathway candidati) fra gli edge significativi.\n\n")
        for idx, subgraph in enumerate(pathway_subgraphs):
            residues = sorted(subgraph.nodes())
            f.write(f"--- Pathway {idx} ({len(residues)} residui, "
                     f"{subgraph.number_of_edges()} contatti) ---\n")
            f.write(f"Residui: {residues}\n")
            f.write("Contatti (resid_i, resid_j, delta_eb):\n")
            for resid_i, resid_j, data in subgraph.edges(data=True):
                f.write(f"    {resid_i} -- {resid_j} : {data['delta_eb']:.4f}\n")
            f.write("\n")

    print(f"    -> Risultati salvati in: {out_dir}/")


# ----------------------------------------------------------------------
# 6. Funzione principale
# ----------------------------------------------------------------------
def main() -> None:
    print("Caricamento dei grafi pesati...")
    g_open = load_weighted_graph(APO_WEIGHTED_GRAPH)
    g_closed = load_weighted_graph(HOLO_WEIGHTED_GRAPH)

    print("Calcolo della Edge Betweenness nei due stati...")
    eb_open = edge_betweenness_by_resid(g_open, EB_WEIGHT_ATTRIBUTE)
    eb_closed = edge_betweenness_by_resid(g_closed, EB_WEIGHT_ATTRIBUTE)

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