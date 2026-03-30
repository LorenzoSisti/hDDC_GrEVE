# GrEVE: GRaph Evolution Visualization and Evaluation

A pipeline developed during my internship at the [Structural Bioinformatics Group at Sapienza University of Rome](https://schubert.bio.uniroma1.it/index.html).

...All computations performed via the HPC CINECA supercomputer Leonardo...

# Running MD simulations

Loading the GROMACS module:

```
module load profile/lifesc
module load gromacs
```

Running the simulation:

```
sbatch gromacs_leonardo_24h_production6.sh 
```

Per strippare una traiettoria dall'acqua e da eventuali idrogeni:

```
gmx trjconv -f md_0_1_noPBC.xtc -s md_0_1.tpr -o lysozyme_Protein.xtc
```

Then:

```
sbatch --dependency=afterany:"38432378" gromacs_leonardo_24h_production7.sh
```

Per convertire un xtc in dcd:

```
pip install mdtraj
mdconvert -o lysozyme_Protein.dcd  lysozyme_Protein.xtc
```
Work in progress...
