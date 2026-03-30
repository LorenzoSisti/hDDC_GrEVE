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
###Runna in shell una volta per ottenere un tpr unico da usare in tutte le productions runs 
#gmx grompp -f step7_production.mdp -c npt4.gro -t npt4.cpt -p topol.top -o prod.tpr -n index_new_new.ndx

#Imposto 4 runs diverse e le concateno 

#1 run
#pin off
#gmx mdrun -deffnm prod1_pinoff -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s prod.tpr
#pin on 
#gmx mdrun -deffnm prod1_pinon -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin on -nstlist 500 -s prod.tpr

#2 run --> using cpt
#gmx mdrun -deffnm prod2 -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s prod.tpr --cpi prod1.cpt 

#3 run 

#gmx mdrun -deffnm prod3 -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s prod.tpr --cpi prod2.cpt 

#4 run

#gmx mdrun -deffnm prod4 -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s prod.tpr --cpi prod3.cpt 

####PROD2 
# Esempio di comando da inserire nel file .sh
gmx mdrun -deffnm prod7 -s prod_1000ns.tpr -cpi prod6.cpt -nstlist 40 -ntomp 8 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -v -maxh 24 -noappend 
