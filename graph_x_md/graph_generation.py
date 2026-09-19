"""
graph_generation.py
====================

Step 1 della pipeline MD -> Teoria dei Grafi (vedi "Mappa Strategica e
Protocollo Operativo", Fase 1-2): costruisce una rete di contatto
NON PESATA (Residue Interaction Network, RIN) a partire da una o piu'
traiettorie di Dinamica Molecolare, usando esclusivamente gli atomi
C-alpha della proteina come nodi del grafo.

Definizioni usate in questo script (concordate con il gruppo di ricerca):
    * NODO    -> un atomo C-alpha per ogni residuo della proteina.
    * CONTATTO (singolo frame) -> due C-alpha la cui distanza euclidea e'
      < CONTACT_CUTOFF (default: 7.0 Angstrom).
    * EDGE (sull'intera traiettoria) -> due nodi sono collegati nel grafo
      finale solo se sono in contatto per PIU' del FREQ_THRESHOLD (default:
      75%) dei frame analizzati.

Questo script produce SOLO un grafo topologico (adiacenza binaria).
La pesatura degli edge basata sulla Correlazione Generalizzata (GC) verra'
aggiunta in uno script successivo, una volta validata questa rete di
contatto.

Il codice segue volutamente lo stile della documentazione ufficiale di:
    * MDAnalysis     -> https://docs.mdanalysis.org
    * python-igraph  -> https://python.igraph.org/en/stable/

Nota su Biopython: non e' necessario in questo step. Il linguaggio di
selezione atomica di MDAnalysis ("protein and name CA") isola gia' i
C-alpha direttamente, cioe' esattamente cio' per cui servirebbe la
gerarchia Structure/Model/Chain/Residue/Atom di Biopython. Restiamo quindi
nello stack MDAnalysis + igraph, come richiesto.

Autore: Gruppo di Ricerca in Biologia Computazionale & Dinamica Molecolare
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import igraph as ig
import MDAnalysis as mda
from MDAnalysis.analysis import distances


# ========================================================================
# CONFIGURAZIONE — modifica qui i path e i parametri, poi lancia
# semplicemente: python graph_generation.py
# ========================================================================

# --- Sistema Apo / Open --------------------------------------------------
APO_TOPOLOGY = "apo.pdb"          # file di topologia (pdb, psf, prmtop, gro, ...)
APO_TRAJECTORY = "apo_concat.xtc"  # traiettoria concatenata, gia' senza acqua

# --- Sistema Holo / Closed ------------------------------------------------
HOLO_TOPOLOGY = "holo.pdb"
HOLO_TRAJECTORY = "holo_concat.xtc"

# --- Parametri del grafo di contatto ---------------------------------------
CONTACT_CUTOFF = 7.0       # Angstrom: distanza sotto la quale c'e' contatto
FREQ_THRESHOLD = 0.75      # frazione di frame minima per mantenere un edge
MIN_SEQ_SEPARATION = 0     # 0 = nessuna esclusione di vicini in sequenza
TRAJECTORY_STEP = 1        # 1 = analizza ogni frame; aumenta per test rapidi

# --- Output -----------------------------------------------------------------
OUTPUT_DIR = "./graphs"    # cartella dove salvare i grafi .graphml / .pkl


# ----------------------------------------------------------------------
# 1. FUNZIONE PRINCIPALE: costruisce un grafo di contatto da una traiettoria
# ----------------------------------------------------------------------
def build_contact_graph(
    topology: str,
    trajectory: str,
    label: str = "system",
    contact_cutoff: float = 7.0,
    freq_threshold: float = 0.75,
    min_seq_separation: int = 0,
    step: int = 1,
) -> ig.Graph:
    """
    Costruisce un grafo di contatto residuo-residuo (non pesato) a partire
    da una traiettoria di MD.

    Parameters
    ----------
    topology : str
        Path al file di topologia (es. .pdb, .psf, .prmtop, .gro, ...),
        compatibile con MDAnalysis.Universe().
    trajectory : str
        Path alla traiettoria (gia' strippata dall'acqua e gia'
        concatenata) (es. .xtc, .dcd, .nc, ...).
    label : str
        Nome leggibile del sistema (es. "apo_open", "holo_closed").
        Viene salvato come attributo a livello di grafo.
    contact_cutoff : float
        Distanza (in Angstrom) al di sotto della quale due C-alpha sono
        considerati "in contatto" in un singolo frame. Default: 7.0 A.
    freq_threshold : float
        Frazione minima di frame (0-1) in cui un contatto deve essere
        osservato affinche' l'edge corrispondente venga mantenuto nel
        grafo finale. Default: 0.75 (cioe' > 75% del tempo di
        simulazione).
    min_seq_separation : int
        Filtro opzionale per escludere i contatti "banali" del backbone
        covalente tra residui vicini in sequenza (|i - j| <=
        min_seq_separation). Di default e' 0 (nessuna esclusione, come
        richiesto): tutte le coppie di residui entro 7 A vengono
        considerate. Parametro lasciato disponibile per eventuali
        raffinamenti futuri (in molte Residue Interaction Network si
        escludono i vicini con |i-j|<=2).
    step : int
        Passo di campionamento della traiettoria (1 = ogni frame). Utile
        per un test rapido su traiettorie molto lunghe.

    Returns
    -------
    igraph.Graph
        Grafo non diretto e non pesato. Ogni vertice corrisponde a un
        atomo C-alpha/residuo e porta gli attributi:
            resid, resname, segid, resindex, node_label
        Ogni edge porta un solo attributo:
            contact_frequency  (frazione di frame in cui il contatto e'
                                 stato osservato)
        Attributi a livello di grafo:
            label, n_frames_analysed, contact_cutoff_angstrom,
            freq_threshold, min_seq_separation
    """

    print(f"\n[{label}] Caricamento Universe...")
    print(f"    topology   = {topology}")
    print(f"    trajectory = {trajectory}")
    u = mda.Universe(topology, trajectory)

    # --- Selezione dei nodi del grafo: solo C-alpha della proteina -----
    ca_atoms = u.select_atoms("protein and name CA")
    n_nodes = ca_atoms.n_atoms
    if n_nodes == 0:
        raise ValueError(
            f"[{label}] Nessun atomo C-alpha trovato con la selezione "
            "'protein and name CA'. Controlla il file di topologia / i "
            "nomi degli atomi."
        )
    print(f"    -> {n_nodes} atomi C-alpha selezionati (= nodi del grafo).")

    # Maschera booleana (n_nodes, n_nodes) per escludere opzionalmente le
    # coppie di residui troppo vicine in sequenza (min_seq_separation).
    resindices = ca_atoms.resindices  # indice di residuo (0-based), uno per atomo
    seq_sep = np.abs(resindices[:, None] - resindices[None, :])
    keep_pair_mask = seq_sep > min_seq_separation
    np.fill_diagonal(keep_pair_mask, False)  # un nodo non e' mai in "contatto" con se stesso

    # --- Accumulo dei contatti lungo la traiettoria ---------------------
    contact_counts = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    n_frames = 0

    for ts in u.trajectory[::step]:
        # distance_array rispetta le condizioni periodiche al contorno
        # quando e' presente una box di simulazione (u.dimensions), come
        # raccomandato nella documentazione MDAnalysis per
        # distances.distance_array.
        dist_matrix = distances.distance_array(
            ca_atoms.positions, ca_atoms.positions, box=u.dimensions
        )
        frame_contacts = (dist_matrix < contact_cutoff) & keep_pair_mask
        contact_counts += frame_contacts
        n_frames += 1

        if n_frames % 100 == 0:
            print(f"    ... processati {n_frames} frame")

    if n_frames == 0:
        raise RuntimeError(f"[{label}] La traiettoria non contiene frame da analizzare.")

    print(f"    -> Completato. Frame totali analizzati: {n_frames}")

    # --- Calcolo frequenza di contatto e soglia per definire gli edge ---
    contact_frequency = contact_counts / n_frames

    # La matrice e' simmetrica: basta lavorare sul triangolo superiore.
    i_idx, j_idx = np.triu_indices(n_nodes, k=1)
    freqs = contact_frequency[i_idx, j_idx]
    edge_mask = freqs > freq_threshold

    edges = list(zip(i_idx[edge_mask].tolist(), j_idx[edge_mask].tolist()))
    edge_freqs = freqs[edge_mask].tolist()

    print(
        f"    -> {len(edges)} edge mantenuti "
        f"(contatto presente in > {freq_threshold:.0%} dei frame)."
    )

    # --- Costruzione dell'oggetto igraph.Graph ---------------------------
    g = ig.Graph(n=n_nodes, edges=edges, directed=False)

    # Attributi dei vertici (uno per nodo, stesso ordine di ca_atoms:
    # il vertice i di igraph corrisponde a ca_atoms[i]).
    g.vs["resid"] = ca_atoms.resids.tolist()
    g.vs["resname"] = ca_atoms.resnames.tolist()
    g.vs["segid"] = ca_atoms.segids.tolist()
    g.vs["resindex"] = ca_atoms.resindices.tolist()
    # Etichetta leggibile tipo "ALA45", comoda per plot/debug.
    g.vs["node_label"] = [
        f"{rn}{ri}" for rn, ri in zip(g.vs["resname"], g.vs["resid"])
    ]

    # Attributo di edge: frazione di tempo di simulazione in cui il
    # contatto e' stato osservato.
    # NOTA: NON e' ancora il peso finale della roadmap (che sara'
    # wij = -log(GCij), calcolato in uno script successivo). Per questo
    # motivo lo teniamo sotto un nome diverso da "weight", per evitare
    # qualsiasi ambiguita' futura.
    g.es["contact_frequency"] = edge_freqs

    # Attributi a livello di grafo (provenienza / bookkeeping).
    g["label"] = label
    g["n_frames_analysed"] = n_frames
    g["contact_cutoff_angstrom"] = contact_cutoff
    g["freq_threshold"] = freq_threshold
    g["min_seq_separation"] = min_seq_separation

    return g


# ----------------------------------------------------------------------
# 2. Funzione di supporto per il salvataggio su disco
# ----------------------------------------------------------------------
def save_graph(g: ig.Graph, output_dir: str, label: str) -> None:
    """
    Salva il grafo su disco in due formati:
        * GraphML (.graphml) -> leggibile, compatibile con Cytoscape/Gephi
        * pickle Python (.pkl) -> preserva l'oggetto igraph.Graph cosi'
          com'e', comodo per i prossimi script della pipeline (pesatura
          degli edge, centralita', community detection, ...).

    Parameters
    ----------
    g : igraph.Graph
        Grafo prodotto da build_contact_graph().
    output_dir : str
        Cartella di destinazione (creata se non esiste).
    label : str
        Usato come nome base dei file (es. "apo_open" -> apo_open.graphml).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphml_path = out_dir / f"{label}.graphml"
    pickle_path = out_dir / f"{label}.pkl"

    g.write_graphml(str(graphml_path))
    g.write_pickle(str(pickle_path))

    print(f"    -> Salvato: {graphml_path}")
    print(f"    -> Salvato: {pickle_path}")


# ----------------------------------------------------------------------
# 3. Funzione principale — usa i parametri definiti nella sezione
#    CONFIGURAZIONE in cima al file.
# ----------------------------------------------------------------------
def main() -> None:
    # I due sistemi della roadmap: Apo/Open e Holo/Closed.
    systems = [
        dict(topology=APO_TOPOLOGY, trajectory=APO_TRAJECTORY, label="apo_open"),
        dict(topology=HOLO_TOPOLOGY, trajectory=HOLO_TRAJECTORY, label="holo_closed"),
    ]

    for system in systems:
        g = build_contact_graph(
            topology=system["topology"],
            trajectory=system["trajectory"],
            label=system["label"],
            contact_cutoff=CONTACT_CUTOFF,
            freq_threshold=FREQ_THRESHOLD,
            min_seq_separation=MIN_SEQ_SEPARATION,
            step=TRAJECTORY_STEP,
        )
        save_graph(g, output_dir=OUTPUT_DIR, label=system["label"])

    print("\nCompletato. Entrambi i grafi di contatto sono stati generati e salvati.")


if __name__ == "__main__":
    main()