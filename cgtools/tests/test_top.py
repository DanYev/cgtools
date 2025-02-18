from cgtools.cgtools.forcefields import martini30rna
from cgtools.cgtools.topology import Topology


def test_top():
    forcefield = martini30rna()
    topol = Topology(forcefield=forcefield, sequence=['A', 'G', 'C', 'U'], )
    topol.process_bb_bonds()
    for bond in topol.bonds:
        print(bond)


if __name__ == "__main__":
    test_top()   