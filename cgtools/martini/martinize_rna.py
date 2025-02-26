"""
Usage: python martinize_rna.py -f ssRNA.pdb -mol rna -elastic yes -ef 100 -el 0.5 -eu 1.2 -os molecule.pdb -ot molecule.itp
"""
import argparse
import cgtools.forge.forcefields as ffs
import cgtools.forge.cgmap as cgmap
from cgtools.forge.topology import Topology
from cgtools.pdbtools import AtomList, pdb2atomlist, pdb2system


def martinize_rna_parser():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="CG Martini FF for RNA")
    parser.add_argument("-f",  required=True, type=str, help="Input PDB file")
    parser.add_argument("-ot", default="molecule.itp", type=str, help="Output topology file (default: molecule.itp)")
    parser.add_argument("-os", default="molecule.pdb", type=str, help="Output CG structure (default: molecule.pdb)")
    parser.add_argument("-ff", default="reg", type=str, help="Force field: regular or polar (reg/pol) (default: reg)")
    parser.add_argument("-mol", default="molecule", type=str, help="Molecule name in the .itp file (default: molecule)")
    parser.add_argument("-merge", default=True, type=bool, help="Merge separate chains if detected (default: True)")
    parser.add_argument("-elastic", default=False, type=bool, help="Add elastic network (default: False)")
    parser.add_argument("-ef", default=200, type=float, help="Elastic network force constant (default: 200 kJ/mol/nm^2)")
    parser.add_argument("-el", default=0.5, type=float, help="Elastic network lower cutoff (default: 0.5 nm)")
    parser.add_argument("-eu", default=1.2, type=float, help="Elastic network upper cutoff (default: 1.2 nm)")    
    parser.add_argument("-p", default='backbone', type=str, help="Output position restraints (no/backbone/all) (default: None)")  
    parser.add_argument("-pf", default=1000, type=float, help="Position restraints force constant (default: 1000 kJ/mol/nm^2)")  
    return parser.parse_args()


def process_chain(chain, ff, start_atom, molname):
    # Mapping
    atoms = cgmap.map_chain(chain, ff, atid=start_atom) # Map residue according to the force-field. Returns list of CG atoms 
    # Topology 
    sequence = [residue.resname for residue in chain] # So far only need sequence for the topology
    top = Topology(forcefield=ff, sequence=sequence, molname=molname)
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
    # Reading options
    options = martinize_rna_parser()
    if options.ff  == 'reg':  # Force field
        ff = ffs.martini30rna()
    if options.ff  == 'pol':
        ff = ffs.martini31rna()
    inpdb = options.f
    molname = options.mol
    # Processing chains
    system = pdb2system(inpdb) 
    cgmap.move_o3(system) # Need to move all O3's to the next residue. Annoying but wcyd
    structure, topologies = AtomList(), [] 
    start_atom = 1
    for chain in system.chains():
        atoms, top = process_chain(chain, ff, start_atom, molname)
        structure.extend(atoms)
        topologies.append(top)
        start_atom += len(atoms)
    structure.write_pdb(options.os) 
    # Finishing topologies
    top = merge_topologies(topologies)
    if options.elastic: 
        top.elastic_network(structure, anames=['BB1', 'BB3',], el=options.el, eu=options.eu, ef=options.ef) 
    top.write_to_itp(options.ot)    

        
   
