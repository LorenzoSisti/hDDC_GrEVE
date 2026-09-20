#!/bin/bash
#SBATCH --job-name=pca_apo_holo
#SBATCH --partition=boost_usr_prod        # Usa 'dcgp_usr_prod' per calcoli CPU (consigliato per PCA)
#SBATCH --account=IscrC_hDDC             # Il tuo account CINECA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=pca_%j.out
#SBATCH --error=pca_%j.err

set -euo pipefail

echo "Job iniziato su $(hostname) alle $(date)"

# --- Moduli ---------------------------------------------------------------
module purge
module load profile/chem-phys
module load gromacs/2024.2--gcc--12.2.0-cuda-12.2
module load python/3.11  

# Se usi un ambiente virtuale (virtualenv/conda) per librerie come MDAnalysis o NumPy, 
# decommenta e adatta la riga sotto (sostituendo il percorso corretto):
# source /leonardo/home/userexternal/lsisti00/venvs/pca_env/bin/activate

source $SLURM_SUBMIT_DIR/pca_env/bin/activate

# --- Percorsi ---------------------------------------------------------------
# NOTA: Verifica che questi percorsi in scratch esistano davvero!
BASE_DIR=/leonardo_scratch/large/userexternal/lsisti00/pca
OUT_DIR=/leonardo_scratch/large/userexternal/lsisti00/pca_results/run_$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR=$SLURM_SUBMIT_DIR

mkdir -p "$OUT_DIR"

# --- Esecuzione --------------------------------------------------------------
python3 "$SCRIPT_DIR/pca_pipeline.py" \
    --base-dir "$BASE_DIR" \
    --out-dir "$OUT_DIR" \
    --ref-system apo_open \
    --compare-systems holo_open holo_closed \
    --group C-alpha \
    --n-eigenvectors 2

echo "Job terminato alle $(date)"
