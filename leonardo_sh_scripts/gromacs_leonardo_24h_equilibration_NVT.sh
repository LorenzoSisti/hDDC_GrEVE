#!/bin/bash
#SBATCH --job-name gromacs 
#SBATCH -N1 --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_hDDC
#SBATCH --partition=boost_usr_prod


module load profile/lifesc
module load gromacs
 
export OMP_NUM_THREADS=8

export GMX_GPU_DD_COMMS=true 
export GMX_GPU_PME_PP_COMMS=true
export GMX_FORCE_UPDATE_DEFAULT_GPU=true

##gmx mdrun  -deffnm adh_pme4 -ntomp 8 -v -maxh 1.0 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s topol.tpr 

######################NVT#######################

#gmx grompp -f NVT_equilibration_step.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -n index.ndx #farlo nella shell direttamente, non nello script 
gmx mdrun -deffnm nvt -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s nvt.tpr
#gmx grompp -f step6.2_equilibration.mdp -c nvt1.gro -r nvt1.gro -t nvt1.cpt -p topol.top -o nvt2.tpr -n index_new_new.ndx
#gmx mdrun -deffnm nvt2 -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s nvt2.tpr



#NVT 1 and 2 
#for i in `seq 1 2`
 #      do
              #x=$(($i-1))
#	      gmx mdrun -deffnm nvt$i -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s nvt$i.tpr
#done



#gmx mdrun -deffnm nvt$i -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s nvt$i.tpr


