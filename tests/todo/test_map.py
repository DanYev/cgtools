from reforge.forge.forcefields import martini30rna
import reforge.forge.cgmap as cgmap
                                 
if __name__ == "__main__":
    ff = martini30rna()
    pdb = 'chain_A.pdb'
    system = cgmap.read_pdb(pdb)   
    cgmap.move_o3(system)
    cgchain = cgmap.map_residues(system, ff, atid=1)
    cgmap.save_pdb(cgchain, fpath='test.pdb')    


