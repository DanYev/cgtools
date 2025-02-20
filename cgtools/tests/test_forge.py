import cgtools.forge.forcefields as ffs
import cgtools.forge.cgmap as cgmap
import copy
from cgtools.forge.topology import Topology
                                 

def process_chain(chain, ff, start_atom):
    # Mapping
    atoms = cgmap.map_residues(chain, ff, atid=start_atom) # Map residue according to the force-field. Returns list of CG atoms 
    # Topology 
    sequence = [residue.resname for residue in chain] # So far only need sequence for the topology
    top = Topology(forcefield=ff, sequence=sequence, )
    top.process_atoms() # Adds itp atom objects to the topology list 
    top.process_bb_bonds() # Adds bb bond objects to the topology list 
    top.process_sc_bonds() # Adds sc bond objects to the topology list 
    return atoms, top


def merge_topologies(topologies):
    top = topologies.pop(0)
    if topologies:
        for new_top in topologies:
            top += new_top
    return top


if __name__ == "__main__":
    ff = ffs.martini30rna()
    pdb = 'dsRNA.pdb'
    system = cgmap.read_pdb(pdb) 
    cgmap.move_o3(system) # Need to move all O3's to the next residue. Annoying but wcyd
    structure, topologies = [], [] 
    start_atom = 1
    for chain in system.chains():
        atoms, top = process_chain(chain, ff, start_atom)
        structure.extend(atoms)
        topologies.append(top)
        start_atom += len(atoms)
    cgmap.save_pdb(structure, fpath='test.pdb') # Saving CG structure    
    top = merge_topologies(topologies)
    top.elastic_network(structure, anames=['BB1', 'BB3',], el=0.5, eu=1.2, ef=200) # Adds sc bond objects to the topology list 
    top.write_itp('test.itp')    

        
   
