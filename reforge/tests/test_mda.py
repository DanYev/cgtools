import MDAnalysis as mda
import reforge.dataio as io

s = 'mdc.pdb' # topology
f = 'mdc.trr' # trajectory
# s = 'ref.pdb'


u = mda.Universe(s, f, in_memory=True)
mask = u.atoms.segids == 'A'
ag = u.atoms[mask]

positions = io.read_positions(u, ag)
print(positions.shape, len(ag))
print(u.trajectory)
