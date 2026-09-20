"""
normal_mode_visualization.py
============================
Visualizzazione 3D per PC1 e PC2 (Putty B-factor e Video Movie).
Compatibile con le ultime versioni di MDAnalysis (v2.x+).
"""

import argparse
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import pca


def process_pc_component(
    u: mda.Universe,
    sel_atoms: mda.AtomGroup,
    pca_analysis: pca.PCA,
    pc_idx: int,
    out_dir: Path,
    n_frames: int = 30,
) -> None:
    pc_num = pc_idx + 1
    print(f"\n--- Elaborazione PC{pc_num} ---")

    # 1. Estrazione componente e rimozione parte immaginaria residua
    if hasattr(pca_analysis, "results"):
        pc_vector = pca_analysis.results.p_components[:, pc_idx]
    else:
        pc_vector = pca_analysis.p_components[:, pc_idx]

    pc_vector = np.real(pc_vector)
    pc_disp = np.linalg.norm(pc_vector.reshape(-1, 3), axis=1)

    # Normalizzazione 0-10 per B-factor PyMOL
    pc_disp_scaled = (pc_disp - pc_disp.min()) / (pc_disp.max() - pc_disp.min()) * 10.0

    # 2. Generazione PDB per figura Putty
    u.trajectory[0]

    if not hasattr(u.atoms, "tempfactors"):
        u.add_TopologyAttr("tempfactors")

    u.atoms.tempfactors = 0.0
    sel_atoms.tempfactors = pc_disp_scaled

    putty_pdb = out_dir / f"protein_pc{pc_num}_putty.pdb"
    u.atoms.write(str(putty_pdb))
    print(f" -> [Putty] File B-factor salvato: {putty_pdb}")

    # 3. Estrazione coordinate medie
    if hasattr(pca_analysis, "results") and hasattr(pca_analysis.results, "mean"):
        mean_positions = pca_analysis.results.mean
    elif hasattr(pca_analysis, "mean") and pca_analysis.mean is not None:
        mean_positions = pca_analysis.mean
    else:
        mean_positions = sel_atoms.positions

    mean_positions = np.real(mean_positions).reshape(-1, 3)

    # 4. Generazione Traiettoria per il Video (Movie)
    transformed = pca_analysis.transform(sel_atoms, n_components=pc_num)
    pc_projections = np.real(transformed[:, pc_idx])

    min_val, max_val = pc_projections.min(), pc_projections.max()
    pc_3d = pc_vector.reshape(-1, 3)

    angles = np.linspace(0, 2 * np.pi, n_frames)
    amplitude = (max_val - min_val) / 2.0
    mid_val = (max_val + min_val) / 2.0

    movie_pdb = out_dir / f"pc{pc_num}_motion_movie.pdb"
    with mda.Writer(str(movie_pdb), sel_atoms.n_atoms) as W:
        for a in angles:
            proj_val = mid_val + amplitude * np.sin(a)
            new_pos = mean_positions + proj_val * pc_3d
            sel_atoms.positions = new_pos
            W.write(sel_atoms)

    print(f" -> [Movie] Traiettoria video salvata: {movie_pdb}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_results_dir = script_dir.parent / "pca_results"

    parser = argparse.ArgumentParser(description="Genera file PDB Putty e Movie per PC1 e PC2.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"Cartella dei risultati PCA (default: {default_results_dir})"
    )
    parser.add_argument("--group", default="name CA", help="Selezione atomica (default: name CA)")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"La cartella risultati non esiste: {results_dir}")

    concat_xtc = list(results_dir.glob("*_concat.xtc"))
    if not concat_xtc:
        raise FileNotFoundError(f"Nessun file *_concat.xtc trovato in {results_dir}")
    xtc_file = concat_xtc[0]

    tpr_candidates = list(results_dir.glob("*.tpr")) + list(results_dir.glob("*.pdb"))
    if not tpr_candidates:
        raise FileNotFoundError(f"Nessun file .tpr/.pdb trovato in {results_dir}")
    top_file = tpr_candidates[0]

    print(f"Cartella lavoro: {results_dir}")
    print(f"Topologia usata: {top_file.name}")
    print(f"Traiettoria usata: {xtc_file.name}")

    u = mda.Universe(str(top_file), str(xtc_file))
    sel_atoms = u.select_atoms(args.group)

    print(f"Calcolo PCA sulle prime 2 componenti principali ({len(sel_atoms)} atomi)...")
    pca_analysis = pca.PCA(u, select=args.group, n_components=2).run()

    for pc_idx in [0, 1]:
        process_pc_component(
            u=u,
            sel_atoms=sel_atoms,
            pca_analysis=pca_analysis,
            pc_idx=pc_idx,
            out_dir=results_dir,
        )

    print("\nCompletato! File generati con successo nella cartella pca_results.")


if __name__ == "__main__":
    main()