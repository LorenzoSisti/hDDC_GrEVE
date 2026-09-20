# Carica entrambi i PDB
load protein_pc1_putty.pdb, pc1_putty
load protein_pc2_putty.pdb, pc2_putty

# Disponi le strutture affiancate
set grid_mode, 1

# Stile Putty e colorazione B-factor per entrambe
hide everything
show cartoon
cartoon putty
spectrum b, blue_white_red

# Uniforma lo spessore minimo/massimo del nastro
set cartoon_putty_transform, 0
set cartoon_putty_scale_max, 3.0

# Generare xtc filtrato per componente principale:
echo -e "3\n3\n3" | gmx anaeig -s apo_open_prod_1000ns.tpr -f apo_open_concat.xtc -v eigenvec.trr -first 2 -last 2 -filt pc2_filtered.xtc

# Per caricare i video: 
load calpha_ref.pdb
load_traj pc1_filtered.xtc, calpha_ref
hide everything
show ribbon, calpha_ref