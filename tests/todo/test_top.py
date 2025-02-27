from reforge import itpio
from reforge.forge.forcefields import martini30rna
from reforge.forge.topology import Topology


def test_top():
    forcefield = martini30rna()
    topol = Topology(forcefield=forcefield, sequence=list('ACGU'), )
    topol.process_atoms()
    topol.process_bb_bonds()
    topol.process_sc_bonds()
    # print(topol.atoms)
    # print(topol.bonds)
    # print(topol.angles)
    # print(topol.dihs)
    # print(topol.cons)
    # print(topol.excls)
    # print(topol.pairs)
    # print(topol.vs3s)

    lines = itpio.format_header()
    lines += itpio.format_moleculetype_section()
    lines += itpio.format_atoms_section(topol.atoms)
    lines += itpio.format_bonded_section('bonds', topol.bonds)
    lines += itpio.format_bonded_section('angles', topol.angles)
    lines += itpio.format_bonded_section('dihedrals', topol.dihs)
    lines += itpio.format_bonded_section('constraints', topol.cons)
    lines += itpio.format_bonded_section('exclusions', topol.excls)
    lines += itpio.format_bonded_section('pairs', topol.pairs)
    lines += itpio.format_bonded_section('virtual_sites3', topol.vs3s)
    lines += itpio.format_posres_section(topol.atoms)
    itpio.write_itp('test.itp', lines)


if __name__ == "__main__":
    test_top()   