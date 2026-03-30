#!/bin/bash
#SBATCH --job-name prod1_dopa_plp 
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
#gmx grompp -f NPT_production_step.mdp -c npt.gro -t npt.cpt -p topol.top -o prod_1000ns.tpr -n index.ndx

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

####PROD1 with pin off 
gmx mdrun -deffnm prod1 -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin off -nstlist 500 -s prod_1000ns.tpr
#gmx mdrun -deffnm prod1_pinon -ntomp 8 -v -maxh 24 -ntmpi 4 -nb gpu -pme gpu -npme 1 -pin on -nstlist 500 -s prod.tpr


#prova run con pin on e pin off durante la produzione 

#produzione 
#..time =24:00:00
#opzione gromacs -maxh 24
#automaticamene gromacs, quando job arriva a 23:59 min, inetrrompe la simulazione e genera un file cpt --> 
#salva i valori di forze, energie, tempo e veclotià per fare restart della simualzioneO

#n scripts --> per ognuno cambio la riga di comando in modo tale che 
#deffnm md1
#deffnm md2 (secondo script)
#uso un unico tpr da 500 ns --> la prima simulazione si interrompe, 
#uso il secondo scirpt usando lo stesso tpr e facedno ripartire la stessa simualzione da dove si è interrotto lo script, usando il cpt file, uso secondo comando diverso di gromcas 
#dal secodno script in poi uso sempre cpt 
#comando cpt --> --cpi md1.cpt file deffnm output md2

#opzione: n scripts che posso concatenare, es 5 scripts in cui ho già md1, md0 per primo scritp eccc, poi uso opzione slurm per sottomettere questi scritp uno appresso 
#all'altro  e fa in modo che il successivo parta solo quando il rpecednete ha finito
#vai alla guida slurm https://wiki.u-gov.it/confluence/display/SCAIUS/6%3A+Scheduler+and+Job+Submission
#> sbatch job1.cmd
 #submitted batch job 100
#> sbatch --dependency=afterok:100 job2.cmd
# submitted batch job 102




#afterany invece di afterok
#logia: se exit status (variabile numerica unix che puo assumere 0 o 1 a secinda che un processo sia finito correttamente o meno) del job precedente, 
#qualsiasi sia exit status, vai avanti  

#opzione2 --> comando 



#COMANDI DI SCRIPT SEPARATI CONCATENATI 

#[sgiordan@login01 input_fiiles_gromacs]$ sbatch gromacs_leonardo_24h_production1.sh 
#Submitted batch job 15527973
#[sgiordan@login01 input_fiiles_gromacs]$ nano gromacs_leonardo_24h_production1.sh
#[sgiordan@login01 input_fiiles_gromacs]$ nano gromacs_leonardo_24h_production1.sh
#[sgiordan@login01 input_fiiles_gromacs]$ sbatch --dependency=afterany:15527973 gromacs_leonardo_24h_production2.sh 
#Submitted batch job 15527992
#[sgiordan@login01 input_fiiles_gromacs]$ sbatch --dependency=afterany:15527992 gromacs_leonardo_24h_production3.sh 
#Submitted batch job 15527996
#[sgiordan@login01 input_fiiles_gromacs]$ sbatch --dependency=afterany:15527996 gromacs_leonardo_24h_production4.sh 
#Submitted batch job 15528004

 #JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
  #        15527973 boost_usr  gromacs sgiordan PD       0:00      1 (Priority)
   #       15528004 boost_usr  gromacs sgiordan PD       0:00      1 (Dependency)
    #      15527996 boost_usr  gromacs sgiordan PD       0:00      1 (Dependency)
     #     15527992 boost_usr  gromacs sgiordan PD       0:00      1 (Dependency)
