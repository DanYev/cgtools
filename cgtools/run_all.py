from system import System

def run_all():
    system = System('test_system', '4zt0.pdb')
    system.prepare_files()
    # system.clean_pdb()
    # system.split_chains()
    # system.get_go_maps()
    # system.martinize_proteins()
    # system.martinize_nucleotides()
    system.make_topology_file()
    # system.make_gro_file()
    # system.solvate()
    system.add_ions()
  
    
if __name__ == '__main__':
    run_all()