#!/usr/bin/env python3
"""
pca_pipeline.py
================

Pipeline per PCA "aggregata" su repliche di dinamica molecolare GROMACS,
pensata per confrontare stati diversi di una proteina (apo/holo, open/closed)
sullo STESSO piano principale (PC1-PC2).

Logica generale
----------------
1. Trova le repliche del sistema di riferimento (default: apo_open*),
   ciascuna con una traiettoria *_nopbc_fit.xtc e un .tpr nella stessa cartella.
2. Concatena le repliche di riferimento (gmx trjcat -cat).
3. Calcola la matrice di covarianza e gli autovettori (gmx covar) sulla
   traiettoria concatenata -> eigenvec.trr definisce il "piano principale".
4. Proietta OGNI singola replica (del sistema di riferimento e di tutti gli
   altri sistemi da confrontare, es. holo_open, holo_closed) su PC1/PC2
   usando SEMPRE lo stesso eigenvec.trr (gmx anaeig -2d).
5. Genera i grafici comparativi (matplotlib), tenendo traccia della replica
   di provenienza di ogni punto.

Assunzioni sulla struttura delle cartelle
------------------------------------------
base_dir/
    apo_open_1/   *_nopbc_fit.xtc  qualche_file.tpr
    apo_open_2/   ...
    apo_open_3/   ...
    holo_open_1/  ...
    holo_open_2/  ...
    holo_open_3/  ...
    holo_closed_1/...
    holo_closed_2/...
    holo_closed_3/...

Ogni cartella deve contenere ESATTAMENTE UN file che termina con
"_nopbc_fit.xtc" ed ESATTAMENTE UN file .tpr. Se la tua organizzazione è
diversa, modifica la funzione `find_replicas`.

Requisiti
---------
- GROMACS nel PATH (variabile GMX_BIN, default "gmx")
- numpy, matplotlib, pandas

Uso tipico
----------
python pca_pipeline.py \
    --base-dir /leonardo_scratch/.../simulazioni \
    --out-dir  /leonardo_scratch/.../pca_results \
    --ref-system apo_open \
    --compare-systems holo_open holo_closed \
    --group C-alpha \
    --n-eigenvectors 2
"""

import argparse
import glob
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # niente display su HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GMX_BIN = os.environ.get("GMX_BIN", "gmx")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pca_pipeline")


# --------------------------------------------------------------------------- #
# Strutture dati
# --------------------------------------------------------------------------- #
#
# Le due classi sotto sono semplici "contenitori" di dati (create con il
# decoratore @dataclass, che genera automaticamente un __init__ senza
# doverlo scrivere a mano). Servono solo per tenere insieme informazioni
# correlate invece di passarsele in giro come tuple o dizionari sciolti.

@dataclass
class Replica:
    """Rappresenta UNA replica: a quale sistema appartiene, il nome della
    cartella (usato come etichetta nei grafici/CSV) e i percorsi dei due
    file GROMACS necessari (traiettoria .xtc e struttura .tpr)."""
    system: str          # es. "apo_open"
    label: str           # es. "apo_open_1" (nome cartella)
    xtc: Path
    tpr: Path


@dataclass
class System:
    """Rappresenta UN sistema (es. 'apo_open') come insieme delle sue
    repliche. 'field(default_factory=list)' crea una lista VUOTA nuova
    per ogni oggetto System (necessario perche' una lista come valore di
    default "normale" verrebbe condivisa per errore tra tutti gli oggetti,
    e' una particolarita' di Python da usare sempre con dataclass)."""
    name: str
    replicas: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Utility per chiamare GROMACS
# --------------------------------------------------------------------------- #

def run(cmd, input_text=None, cwd=None):
    """
    Wrapper attorno a subprocess.run per lanciare i comandi gmx.

    Parametri di subprocess.run usati sotto:
    - list(map(str, cmd)): 'cmd' e' una lista che puo' contenere sia
      stringhe che oggetti Path o numeri (es. [GMX_BIN, "trjcat", "-f",
      Path("file.xtc"), "-last", 2]). subprocess vuole SOLO stringhe,
      quindi convertiamo ogni elemento a str() prima di eseguire.
    - input=input_text: alcuni tool gmx (covar, anaeig, make_ndx) fanno
      domande interattive a schermo tipo "Select group:". Passando qui
      una stringa (es. "3\n" oppure "q\n") simuliamo la digitazione da
      tastiera + invio, cosi' lo script non resta bloccato in attesa.
    - capture_output=True: cattura sia stdout che stderr del comando
      invece di stamparli a schermo, cosi' possiamo leggerli in Python
      (es. per fare il parsing dei gruppi in get_group_index) o loggarli
      solo in caso di errore.
    - text=True: restituisce stdout/stderr come stringhe di testo
      (invece di bytes grezzi), piu' comodo da maneggiare.

    Ritorna (stdout, stderr) come stringhe; se il comando fallisce
    (returncode diverso da 0) stampa l'errore nel log e interrompe
    l'esecuzione sollevando un'eccezione.
    """
    log.debug("CMD: %s", " ".join(map(str, cmd)))
    result = subprocess.run(
        list(map(str, cmd)),
        input=input_text,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        log.error("Comando fallito: %s", " ".join(map(str, cmd)))
        log.error("STDOUT:\n%s", result.stdout[-3000:])
        log.error("STDERR:\n%s", result.stderr[-3000:])
        raise RuntimeError(f"Comando GROMACS fallito: {cmd[0]}")
    return result.stdout, result.stderr


def get_group_index(tpr: Path, group_name: str, workdir: Path) -> int:
    """
    Trova l'indice numerico di un gruppo (es. 'C-alpha') per un dato .tpr,
    interrogando gmx make_ndx invece di assumere un indice fisso.

    Perche' non possiamo usare un indice fisso (es. "3" per C-alpha)?
    ----------------------------------------------------------------
    GROMACS genera l'elenco dei gruppi "di default" (System, Protein,
    C-alpha, Backbone, ...) in base al CONTENUTO del sistema. Se un
    sistema ha dei residui extra (es. il ligando PLP in holo_open),
    GROMACS puo' creare gruppi aggiuntivi (es. "PLP", "Protein_PLP")
    che spostano la numerazione dei gruppi successivi. Quindi lo
    stesso nome "C-alpha" potrebbe avere indice 3 in un sistema e
    indice 4 in un altro. Per essere sicuri, chiediamo sempre a
    GROMACS l'elenco dei gruppi per QUEL preciso .tpr e cerchiamo il
    nome che ci interessa, invece di indovinare il numero.

    Come funziona in pratica
    -------------------------
    1. Lanciamo `gmx make_ndx -f sistema.tpr -o file_temporaneo.ndx`
       e gli mandiamo in input "q\n" (il tasto 'q' + invio), che
       significa "esci subito senza creare gruppi custom". Il file
       .ndx temporaneo non ci interessa: ci interessa solo il testo
       che make_ndx stampa a schermo (stdout) PRIMA di chiedere input,
       che contiene l'elenco di tutti i gruppi disponibili.
    2. Quell'elenco ha righe con questo formato tipico:

           3 C-alpha             :   410 atoms
           4 Backbone            :  1230 atoms

       cioe': <indice> <nome_gruppo> : <numero_atomi> atoms
    3. Usiamo un'espressione regolare (regex) per estrarre da ogni
       riga l'indice e il nome del gruppo, poi confrontiamo il nome
       con quello cercato (group_name).
    """
    tmp_ndx = workdir / f"_tmp_{tpr.stem}.ndx"
    stdout, _ = run([GMX_BIN, "make_ndx", "-f", tpr, "-o", tmp_ndx], input_text="q\n")
    tmp_ndx.unlink(missing_ok=True)  # non ci serve il file .ndx, solo il testo stampato sopra

    # --- Spiegazione della regex, pezzo per pezzo -----------------------
    #
    #   r"^\s*(\d+)\s+([A-Za-z0-9_\-\.]+)\s*:\s*(\d+)\s+atoms"
    #
    #   ^\s*          inizio riga, seguito da eventuali spazi (le righe di
    #                 make_ndx sono indentate con spazi variabili)
    #   (\d+)         GRUPPO 1: uno o piu' cifre -> e' l'INDICE del gruppo
    #                 (es. "3"). Le parentesi tonde "catturano" il testo
    #                 trovato cosi' da poterlo riusare dopo con m.group(1)
    #   \s+           uno o piu' spazi (separatore tra indice e nome)
    #   ([A-Za-z0-9_\-\.]+)
    #                 GRUPPO 2: il NOME del gruppo. Puo' contenere lettere,
    #                 numeri, underscore "_", trattino "-" o punto "."
    #                 (es. "C-alpha", "Protein_PLP"). Catturato per poterlo
    #                 confrontare dopo con m.group(2)
    #   \s*:\s*       i due punti ":" che separano nome e conteggio atomi,
    #                 con eventuali spazi prima/dopo (es. "C-alpha   :  410")
    #   (\d+)         GRUPPO 3: il numero di atomi nel gruppo (non ci
    #                 serve per la logica, ma completa il pattern della riga
    #                 cosi' da matchare SOLO le righe con questo formato
    #                 esatto e scartare righe di intestazione o vuote)
    #   \s+atoms      la parola "atoms" che chiude la riga
    #
    # In breve: il pattern riconosce SOLO righe nel formato
    # "<numero> <nome> : <numero> atoms" e ignora tutto il resto
    # dell'output di make_ndx (intestazioni, righe vuote, messaggi vari).
    pattern = re.compile(r"^\s*(\d+)\s+([A-Za-z0-9_\-\.]+)\s*:\s*(\d+)\s+atoms")

    # Scorriamo l'output di make_ndx riga per riga (stdout.splitlines()
    # spezza il testo in una lista di righe, una per elemento)
    for line in stdout.splitlines():
        # pattern.match(line) prova ad applicare la regex a QUESTA riga.
        # Se la riga ha il formato atteso, 'm' e' un oggetto "Match" con i
        # gruppi catturati; se la riga non combacia (es. e' vuota, oppure
        # e' un messaggio di make_ndx non pertinente), 'm' e' None.
        m = pattern.match(line)

        # Controlliamo due cose insieme:
        #  - "m" non e' None (cioe' la riga aveva il formato giusto)
        #  - il nome catturato (m.group(2), es. "C-alpha") e' UGUALE
        #    al nome del gruppo che stiamo cercando (group_name)
        if m and m.group(2) == group_name:
            # m.group(1) e' l'indice catturato come STRINGA (es. "3");
            # lo convertiamo a intero per poterlo usare come indice
            # numerico nei comandi gmx covar / gmx anaeig successivi
            idx = int(m.group(1))
            log.info("Gruppo '%s' trovato con indice %d in %s", group_name, idx, tpr.name)
            return idx  # appena trovato il gruppo giusto, usciamo dalla funzione

    # Se il ciclo for finisce senza mai fare "return", vuol dire che
    # nessuna riga aveva il nome cercato: il gruppo non esiste in questo
    # .tpr (es. nome scritto male, maiuscole/minuscole diverse, o il
    # gruppo semplicemente non e' definito per questo sistema).
    raise RuntimeError(
        f"Gruppo '{group_name}' non trovato in {tpr}. "
        f"Controlla il nome esatto del gruppo (case-sensitive)."
    )


# --------------------------------------------------------------------------- #
# Discovery delle repliche
# --------------------------------------------------------------------------- #

def find_replicas(base_dir: Path, system_prefix: str) -> System:
    """
    Cerca tutte le cartelle base_dir/<system_prefix>* e, per ciascuna,
    il file *_nopbc_fit.xtc e il file .tpr (priorità a prod_1000ns.tpr).
    """
    sys_obj = System(name=system_prefix)
    dirs = sorted(glob.glob(str(base_dir / f"{system_prefix}*")))
    if not dirs:
        raise FileNotFoundError(f"Nessuna cartella trovata per pattern '{system_prefix}*' in {base_dir}")

    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        xtc_candidates = sorted(d.glob("*_nopbc_fit.xtc"))
        
        # --- MODIFICA QUI: Cerchiamo prima specificamente prod_1000ns.tpr ---
        specific_tpr = d / "prod_1000ns.tpr"
        if specific_tpr.exists():
            tpr_candidates = [specific_tpr]
        else:
            tpr_candidates = sorted(d.glob("*.tpr"))
        # --------------------------------------------------------------------

        if len(xtc_candidates) != 1:
            log.warning("Skip %s: trovati %d file *_nopbc_fit.xtc (atteso 1)", d, len(xtc_candidates))
            continue
        if len(tpr_candidates) != 1:
            log.warning("Skip %s: trovati %d file .tpr (atteso 1). File trovati: %s", d, len(tpr_candidates), [f.name for f in tpr_candidates])
            continue

        sys_obj.replicas.append(
            Replica(system=system_prefix, label=d.name, xtc=xtc_candidates[0], tpr=tpr_candidates[0])
        )

    if not sys_obj.replicas:
        raise FileNotFoundError(f"Nessuna replica valida trovata per '{system_prefix}'")

    log.info("Sistema '%s': trovate %d repliche -> %s",
              system_prefix, len(sys_obj.replicas), [r.label for r in sys_obj.replicas])
    return sys_obj

# --------------------------------------------------------------------------- #
# Step GROMACS
# --------------------------------------------------------------------------- #

def concat_replicas(system: System, out_dir: Path) -> Path:
    out_xtc = out_dir / f"{system.name}_concat.xtc"
    cmd = [GMX_BIN, "trjcat", "-f", *[r.xtc for r in system.replicas], "-o", out_xtc, "-cat"]
    run(cmd)
    log.info("Concatenazione completata: %s", out_xtc)
    return out_xtc


def compute_covar(concat_xtc: Path, ref_tpr: Path, group_idx: int, out_dir: Path, n_eigenvectors: int):
    eigenval = out_dir / "eigenval.xvg"
    eigenvec = out_dir / "eigenvec.trr"
    covapic = out_dir / "covapic.xpm"
    cmd = [
        GMX_BIN, "covar",
        "-f", concat_xtc,
        "-s", ref_tpr,
        "-o", eigenval,
        "-v", eigenvec,
        "-xpma", covapic,
        "-last", n_eigenvectors,
    ]
    # gmx covar chiede: gruppo per il fit, poi gruppo per l'analisi
    run(cmd, input_text=f"{group_idx}\n{group_idx}\n")
    log.info("Autovettori calcolati: %s", eigenvec)
    return eigenvec, eigenval


def project_2d(xtc: Path, tpr: Path, eigenvec: Path, group_idx: int, out_dir: Path, tag: str) -> Path:
    out_xvg = out_dir / f"proj2d_{tag}.xvg"
    cmd = [
        GMX_BIN, "anaeig",
        "-f", xtc,
        "-s", tpr,
        "-v", eigenvec,
        "-first", 1,
        "-last", 2,
        "-2d", out_xvg,
    ]

    # Controlla se esiste un file index personalizzato nella cartella della replica
    custom_ndx = xtc.parent / "index_custom.ndx"
    if custom_ndx.exists():
        cmd.extend(["-n", str(custom_ndx)])
        # GROMACS accetta anche il nome del gruppo come input di testo!
        group_input = "C-alpha_shared"
    else:
        group_input = str(group_idx)

    run(cmd, input_text=f"{group_input}\n{group_input}\n{group_input}\n")
    return out_xvg


def read_xvg_2d(xvg_path: Path) -> np.ndarray:
    """Legge un file .xvg a 2 colonne (PC1, PC2), ignorando header/commenti."""
    data = []
    with open(xvg_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "@")):
                continue
            parts = line.split()
            data.append((float(parts[0]), float(parts[1])))
    return np.array(data)


# --------------------------------------------------------------------------- #
# Pipeline principale
# --------------------------------------------------------------------------- #

def run_pipeline(base_dir: Path, out_dir: Path, ref_system: str, compare_systems: list,
                  group_name: str, n_eigenvectors: int):

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discovery
    ref = find_replicas(base_dir, ref_system)
    others = [find_replicas(base_dir, s) for s in compare_systems]
    all_systems = [ref] + others

    # 2. Concatenazione del sistema di riferimento
    concat_xtc = concat_replicas(ref, out_dir)

    # 3. Autovettori dal sistema di riferimento concatenato
    ref_tpr_for_covar = ref.replicas[0].tpr
    group_idx_ref = get_group_index(ref_tpr_for_covar, group_name, out_dir)
    eigenvec, eigenval = compute_covar(concat_xtc, ref_tpr_for_covar, group_idx_ref, out_dir, n_eigenvectors)

    # 4. Proiezione di OGNI replica di OGNI sistema sullo stesso piano
    projections = {}  # {system_name: {replica_label: np.ndarray[N,2]}}
    for sys_obj in all_systems:
        projections[sys_obj.name] = {}
        for rep in sys_obj.replicas:
            group_idx = get_group_index(rep.tpr, group_name, out_dir)
            tag = f"{sys_obj.name}_{rep.label}"
            xvg = project_2d(rep.xtc, rep.tpr, eigenvec, group_idx, out_dir, tag)
            projections[sys_obj.name][rep.label] = read_xvg_2d(xvg)
            log.info("Proiettato %s (%d frame)", tag, len(projections[sys_obj.name][rep.label]))

    # 5. Salvataggio dati numerici aggregati in CSV (utile per riusare in altre analisi)
    save_csv(projections, out_dir / "all_projections.csv")

    # 6. Grafici
    make_plots(projections, ref_system, compare_systems, out_dir)

    log.info("Pipeline completata. Output in: %s", out_dir)


def save_csv(projections, out_csv: Path):
    """
    Trasforma il dizionario annidato 'projections' (system -> replica -> array
    Nx2 di PC1/PC2) in una singola tabella "lunga" (una riga per frame) e la
    salva come CSV. Questo formato "long" e' comodo per riusare i dati dopo
    in pandas/seaborn/R senza dover riscrivere la logica di parsing.
    """
    rows = []
    for system_name, reps in projections.items():
        for rep_label, arr in reps.items():
            # 'arr' e' un array numpy Nx2 (N frame, 2 colonne: PC1 e PC2).
            # enumerate(arr) ci da' sia l'indice di riga (i, cioe' il numero
            # di frame) sia la coppia (pc1, pc2) di quella riga, cosi'
            # possiamo "spacchettare" i due valori direttamente nel ciclo.
            for i, (pc1, pc2) in enumerate(arr):
                rows.append({"system": system_name, "replica": rep_label,
                             "frame": i, "PC1": pc1, "PC2": pc2})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("Dati numerici salvati in %s", out_csv)


def make_plots(projections, ref_system: str, compare_systems: list, out_dir: Path):
    plt.rcParams.update({"font.size": 11})

    def scatter_system(ax, system_name, alpha=0.5, s=6, palette=None):
        """Disegna sull'asse 'ax' un punto scatter per OGNI replica del
        sistema, ciascuna con la propria etichetta in legenda (cosi' nei
        grafici si distingue visivamente la replica 1 dalla 2 dalla 3)."""
        reps = projections[system_name]
        for i, (label, arr) in enumerate(reps.items()):
            color = palette[i] if palette else None
            ax.scatter(arr[:, 0], arr[:, 1], s=s, alpha=alpha, label=label, color=color)

    # --- Figura 1: sistema di riferimento da solo, colorato per replica ---
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter_system(ax, ref_system)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{ref_system}: repliche proiettate sul proprio spazio PC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_{ref_system}_only.png", dpi=300)
    plt.close(fig)

    # --- Figure 2..N: sistema di riferimento (grigio, sfondo) + ciascun sistema di confronto ---
    for cmp_system in compare_systems:
        fig, ax = plt.subplots(figsize=(6, 5))

        # Disegniamo le repliche del sistema di riferimento come sfondo
        # grigio chiaro. Le disegniamo SEPARATAMENTE per ogni replica (il
        # ciclo for) ma NON passiamo 'label=' qui apposta: se lo facessimo,
        # la legenda mostrerebbe 3 voci grigie identiche ("apo_open_1",
        # "apo_open_2", "apo_open_3"), una per ogni replica, che sarebbe
        # ridondante visto che nel grafico di confronto non ci interessa
        # distinguere le repliche del riferimento, solo mostrarlo come sfondo.
        for arr in projections[ref_system].values():
            ax.scatter(arr[:, 0], arr[:, 1], s=6, alpha=0.25, color="lightgray")

        # Trucco per avere UNA SOLA voce "apo_open" in legenda per lo sfondo:
        # disegniamo uno scatter con array VUOTI ([], []) - quindi invisibile,
        # non aggiunge nessun punto al grafico - ma con 'label' impostato.
        # matplotlib aggiunge comunque una voce in legenda per ogni elemento
        # disegnato, anche se non ha punti. E' il modo standard in
        # matplotlib per controllare manualmente cosa appare in legenda.
        ax.scatter([], [], color="lightgray", label=ref_system)

        # Ora sovrapponiamo il sistema di confronto (holo_open o
        # holo_closed) con colori/trasparenza diversi, distinguendo le
        # sue repliche (qui invece VOGLIAMO vedere le repliche separate).
        scatter_system(ax, cmp_system, alpha=0.6, s=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{cmp_system} vs {ref_system} (spazio PC di {ref_system})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_{cmp_system}_vs_{ref_system}.png", dpi=300)
        plt.close(fig)

    # --- Figura finale: confronto complessivo apo_open / holo_open / holo_closed ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Assegnamo un colore fisso per sistema (non per replica): qui vogliamo
    # che tutte e 3 le repliche di holo_open abbiano lo STESSO colore, per
    # confrontare a colpo d'occhio i tre sistemi nel loro insieme.
    colors_by_system = {
        ref_system: "tab:gray",
    }
    palette_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, s in enumerate(compare_systems):
        # "i % len(palette_cycle)" fa si' che, se ci fossero piu' sistemi di
        # confronto dei colori disponibili, la palette "ricicli" dall'inizio
        # invece di dare errore (modulo = resto della divisione).
        colors_by_system[s] = palette_cycle[i % len(palette_cycle)]

    for system_name in [ref_system] + compare_systems:
        # projections[system_name] e' un dizionario {replica: array Nx2}.
        # .values() prende solo gli array (scartando le etichette delle
        # repliche) e list(...) li mette in una lista di array.
        # np.vstack impila verticalmente tutti questi array uno sotto
        # l'altro: se replica1 ha 500 frame e replica2 ne ha 500, il
        # risultato e' un unico array (1000, 2) con TUTTI i frame del
        # sistema insieme, perche' qui non ci interessa piu' distinguere
        # la replica di provenienza, solo il sistema nel suo complesso.
        pts = np.vstack(list(projections[system_name].values()))
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.35,
                   color=colors_by_system[system_name], label=system_name)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Confronto campionamento conformazionale\n(spazio PC definito da " + ref_system + ")")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_overview_all_systems.png", dpi=300)
    plt.close(fig)

    log.info("Grafici salvati in %s", out_dir)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", required=True, type=Path, help="Cartella che contiene le sottocartelle dei sistemi")
    p.add_argument("--out-dir", required=True, type=Path, help="Cartella di output per risultati e grafici")
    p.add_argument("--ref-system", default="apo_open", help="Prefisso del sistema di riferimento (default: apo_open)")
    p.add_argument("--compare-systems", nargs="+", default=["holo_open", "holo_closed"],
                   help="Prefissi dei sistemi da proiettare sullo spazio PC del riferimento")
    p.add_argument("--group", default="C-alpha",
                   help="Nome del gruppo GROMACS da usare per fit/analisi PCA (default: C-alpha)")
    p.add_argument("--n-eigenvectors", type=int, default=2,
                   help="Numero di autovettori da calcolare con gmx covar (default: 2, minimo richiesto per -2d)")
    return p.parse_args()


def main():
    args = parse_args()
    run_pipeline(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        ref_system=args.ref_system,
        compare_systems=args.compare_systems,
        group_name=args.group,
        n_eigenvectors=args.n_eigenvectors,
    )


if __name__ == "__main__":
    main()
