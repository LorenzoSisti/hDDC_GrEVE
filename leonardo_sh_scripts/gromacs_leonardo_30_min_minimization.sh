#!/bin/bash
#SBATCH --job-name gromacs 
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=IscrC_hDDC
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg

module load profile/lifesc
module load gromacs

#ntomp 
export OMP_NUM_THREADS=4

export GMX_GPU_DD_COMMS=true 
export GMX_GPU_PME_PP_COMMS=true
export GMX_FORCE_UPDATE_DEFAULT_GPU=true

#command line del templato, non valida
##gmx mdrun  -deffnm adh_pme4 -ntomp 8 -v -maxh 1.0 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s topol.tpr 
#farlo nella shell direttamente, non nello script
#gmx grompp -f energy_minimization_step.mdp -c step5_input.gro -p topol.top -o em.tpr -r step5_input.gro

#MINIMIZZAZIONE 

#script per minimizzaizone, qos da 30 min, ntasks-per-node=1, cpus per task=4, gres=gpu:1, time=30:00
#gmx mdrun -deffnm em -ntomp 4 -v -maxh 0.5 -ntmpi 1 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s em.tpr
gmx mdrun -deffnm em -ntomp 4 -v -maxh 0.5 -ntmpi 1 -pin off -s em.tpr
#nstlist non serve per minimizzazione, ma comuqnue lo mettiamo




