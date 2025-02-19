from cgtools.pdbtools import PDBParser
from cgtools.forge.forcefields import martini30rna


def read_pdb(pdb_path):
    parser = PDBParser(pdb_path)
    system = parser.parse()
    print(system)  # System summary
    return system


def iterate_residues(system):
    """
    Generator that yields each residue from a nested system of models, chains, and residues.
    Args:
        system (iterable): An iterable of models, where each model is an iterable of chains,
                           and each chain is an iterable of residues.
    Yields:
        Residue: Each residue in the system.
    """
    for model in system:
        for chain in model:
            for residue in chain:
                yield residue


if __name__ == "__main__":
    pdb = 'chain_A.pdb'
    system = read_pdb(pdb)   
    ff = martini30rna()
    for residue in iterate_residues(system):
        resmap = ff.mapping[residue.resname]
        print(residue.atoms())
