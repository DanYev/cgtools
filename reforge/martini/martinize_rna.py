#!/usr/bin/env python3
"""
Usage: python martinize_rna.py -f ssRNA.pdb -mol rna -elastic yes -ef 100 -el 0.5 -eu 1.2 
-os molecule.pdb -ot molecule.itp

This script processes an all-atom RNA structure and returns coarse-grained topology in the 
GROMACS' .itp format and coarse-grained PDB.
It parses command-line arguments, processes each chain of the input PDB,
maps them to coarse-grained representations, merges the resulting topologies,
optionally applies an elastic network, and writes the output ITP file.
"""

import argparse
from reforge.forge.forcefields import Martini30RNA
from reforge.forge import cgmap
from reforge.forge.topology import Topology
from reforge.pdbtools import AtomList, pdb2system

def martinize_rna_parser():
    """Parse command-line arguments for RNA coarse-graining."""
    parser = argparse.ArgumentParser(description="CG Martini FF for RNA")
    parser.add_argument("-f", required=True, type=str, help="Input PDB file")
    parser.add_argument(
        "-ot",
        default="molecule.itp",
        type=str,
        help="Output topology file (default: molecule.itp)",
    )
    parser.add_argument(
        "-os",
        default="molecule.pdb",
        type=str,
        help="Output CG structure (default: molecule.pdb)",
    )
    parser.add_argument(
        "-ff",
        default="reg",
        type=str,
        help="Force field: regular or polar (reg/pol) (default: reg)",
    )
    parser.add_argument(
        "-mol",
        default="molecule",
        type=str,
        help="Molecule name in the .itp file (default: molecule)",
    )
    parser.add_argument(
        "-merge",
        default=True,
        type=bool,
        help="Merge separate chains if detected (default: True)",
    )
    parser.add_argument(
        "-elastic",
        default=False,
        type=bool,
        help="Add elastic network (default: False)",
    )
    parser.add_argument(
        "-ef",
        default=200,
        type=float,
        help="Elastic network force constant (default: 200 kJ/mol/nm^2)",
    )
    parser.add_argument(
        "-el",
        default=0.5,
        type=float,
        help="Elastic network lower cutoff (default: 0.5 nm)",
    )
    parser.add_argument(
        "-eu",
        default=1.2,
        type=float,
        help="Elastic network upper cutoff (default: 1.2 nm)",
    )
    parser.add_argument(
        "-p",
        default="backbone",
        type=str,
        help="Output position restraints (no/backbone/all) (default: None)",
    )
    parser.add_argument(
        "-pf",
        default=1000,
        type=float,
        help="Position restraints force constant (default: 1000 kJ/mol/nm^2)",
    )
    return parser.parse_args()


def process_chain(_chain, _ff, _start_idx, _mol_name):
    """
    Process an individual RNA chain: map it to coarse-grained representation and
    generate a topology.

    Args:
        chain (iterable): An RNA chain from the parsed system.
        ff: Force field object.
        start_idx (int): Starting atom index for mapping.
        mol_name (str): Molecule name.

    Returns:
        tuple: (cg_atoms, chain_topology)
    """
    _cg_atoms = cgmap.map_chain(_chain, _ff, atid=_start_idx)
    sequence = [res.resname for res in _chain]
    chain_topology = Topology(forcefield=_ff, sequence=sequence, molname=_mol_name)
    chain_topology.process_atoms()
    chain_topology.process_bb_bonds()
    chain_topology.process_sc_bonds()
    return _cg_atoms, chain_topology


def merge_topologies(top_list):
    """
    Merge multiple Topology objects into one.

    Args:
        top_list (list): List of Topology objects.

    Returns:
        Topology: The merged Topology.
    """
    _merged_topology = top_list.pop(0)
    for new_top in top_list:
        _merged_topology += new_top
    return _merged_topology


if __name__ == "__main__":
    options = martinize_rna_parser()
    if options.ff == "reg":
        ff = Martini30RNA()
    else:
        raise ValueError(f"Unsupported force field option: {options.ff}")
    inpdb = options.f
    mol_name = options.mol
    system = pdb2system(inpdb)
    cgmap.move_o3(system)  # Adjust O3 atoms as required
    structure = AtomList()
    topologies = []
    start_idx = 1
    for chain in system.chains():
        cg_atoms, chain_top = process_chain(chain, ff, start_idx, mol_name)
        structure.extend(cg_atoms)
        topologies.append(chain_top)
        start_idx += len(cg_atoms)
    structure.write_pdb(options.os)
    merged_topology = merge_topologies(topologies)
    if options.elastic:
        merged_topology.elastic_network(
            structure,
            anames=["BB1", "BB3"],
            el=options.el,
            eu=options.eu,
            ef=options.ef,
        )
    merged_topology.write_to_itp(options.ot)
