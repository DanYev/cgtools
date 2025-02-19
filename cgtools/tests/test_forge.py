import cgtools.forge.forcefields as ffs
import cgtools.forge.cgmap as cgmap
import copy
from cgtools.forge.topology import Topology
                                 
if __name__ == "__main__":
    ff = ffs.martini30rna()
    pdb = 'chain_A.pdb'
    # system = cgmap.read_pdb(pdb) 
    # # Mapping the atomic system  
    # cgmap.move_o3(system) # Need to move all O3's to the next residue. Annoying but wcyd  
    # cgchain = cgmap.map_residues(system, ff, atid=1) # Map residue according to the force-field. Returns list of CG atoms 
    # cgmap.save_pdb(cgchain, fpath='test.pdb') # Saving CG structure
    # Topology 
    # sequence = [residue.resname for residue in system.residues()] # So far only need sequence for the topology
    sequence=list('ACGU')
    top = Topology(forcefield=ff, sequence=sequence, )
    top.process_atoms() # Adds itp atom objects to the topology list 
    top.process_bb_bonds() # Adds bb bond objects to the topology list 
    top.process_sc_bonds() # Adds sc bond objects to the topology list 
    # top.write_itp('test.itp')
    new_top = copy.deepcopy(top)
    merged_top = top + new_top
    merged_top.write_itp('test.itp')

        
   
