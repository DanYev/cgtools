import importlib.resources
import os
import sys
from cgtools import itpio


forcefields = ['martini30rna','martini31nucleic']
rna_system = 'test'

# Split each argument in a list                                               
def nsplit(*x):                                                               
    return [i.split() for i in x]  

###################################
## FORCE FIELDS ## 
###################################

class NucleicForceField:
     
    @staticmethod    
    def read_itp(resname, directory, mol, version):
        # itpdir = os.path.abspath(f'/scratch/dyangali/cgtools/forge/forcefields/{directory}')
        itpdir = importlib.resources.files('cgtools') / 'forge' / 'forcefields' 
        file = os.path.join(itpdir, f'{directory}', f'{mol}_{resname}_{version}.itp')
        itp_data = itpio.read_itp(file)
        return itp_data
        
    @staticmethod    
    def itp_to_indata(itp_data):
    # Bond must be a list of tuples [(connectivity), (parameters), (comment)]
        sc_bonds = itp_data['bonds']
        sc_angles = itp_data['angles']
        sc_dihs = itp_data['dihedrals']
        sc_cons = itp_data['constraints']
        sc_excls = itp_data['exclusions']
        sc_pairs = itp_data['pairs']
        sc_vs3s = itp_data['virtual_sites3']
        return sc_bonds, sc_angles, sc_dihs, sc_cons, sc_excls, sc_pairs, sc_vs3s

    @staticmethod    
    def parameters_by_resname(resnames, directory, mol, version):
        params = []
        for resname in resnames:
            itp_data = NucleicForceField.read_itp(resname, directory, mol, version)
            param = NucleicForceField.itp_to_indata(itp_data) 
            params.append(param)
        return dict(zip(resnames, params)) 


    def __init__(self, directory, mol, version):
        # FF location
        self.directory = directory
        self.mol = mol
        self.version = version
        # Dictionary with SC params for each residue
        self.resdict = self.parameters_by_resname(self.resnames, directory, mol, version)
        # Elastic network
        self.elastic_network = False # By default use an elastic network  
        self.el_bond_type = 6 # Elastic networks bond shouldn't lead to exclusions (type 6) 

    def sc_bonds(self, resname):
        return self.resdict[resname][0]

    def sc_angles(self, resname):
        return self.resdict[resname][1]

    def sc_dihs(self, resname):
        return self.resdict[resname][2]

    def sc_cons(self, resname):
        return self.resdict[resname][3]

    def sc_excls(self, resname):
        return self.resdict[resname][4]

    def sc_pairs(self, resname):
        return self.resdict[resname][5]

    def sc_vs3s(self, resname):
        return self.resdict[resname][6]

    def sc_blist(self, resname): # list with all bonded parameters
        res = [self.sc_bonds(resname), self.sc_angles(resname), self.sc_dihs(resname), self.sc_cons(resname), 
            self.sc_excls(resname), self.sc_pairs(resname), self.sc_vs3s(resname)]
        return res
    

class martini30rna(NucleicForceField):
    
    resnames = ['A', 'C', 'G', 'U']

    # FF mapping
    bb_mapping = {
                "BB1":  ("P", "OP1", "OP2", "O5'", "O3'", "O1P", "O2P"), 
                "BB2":  ("C5'", "1H5'", "2H5'", "H5'", "H5''", "C4'", "H4'", "O4'", "C3'", "H3'"), 
                "BB3":  ("C1'", "C2'", "O2'", "O4'")}
    a_mapping = {
                "SC1": ("N9", "C8", "H8"), 
                "SC2": ("N3", "C4"), 
                "SC3": ("N1", "C2", "H2"), 
                "SC4": ("N6", "C6", "H61", "H62"), 
                "SC5": ("N7", "C5")}
    c_mapping = {
                "SC1": ("N1", "C5", "C6"), 
                "SC2": ("C2", "O2"), 
                "SC3": ("N3"), 
                "SC4": ("N4", "C4", "H41", "H42")}
    g_mapping = {
                "SC1": ("C8", "H8", "N9"), 
                "SC2": ("C4", "N3"), 
                "SC3": ("C2", "N2", "H21", "H22"), 
                "SC4": ("N1"), 
                "SC5": ("C6", "O6"), 
                "SC6": ("C5", "N7")}          
    u_mapping = {
                "SC1": ("N1", "C5", "C6"), 
                "SC2": ("C2", "O2"), 
                "SC3": ("N3"), 
                "SC4": ("C4", "O4")}
    mapping = {
        "A":  {**bb_mapping, **a_mapping},
        "C":  {**bb_mapping, **c_mapping},
        "G":  {**bb_mapping, **g_mapping}, 
        "U":  {**bb_mapping, **u_mapping},
    }
    
    # NucleicForceField.update_non_standard_mapping(mapping)
    
    def __init__(self, directory='rna_reg', mol=rna_system, version='new'):
        super().__init__(directory, mol, version)
        self.name = 'martini30rna' 
     
        ##################
        # RNA PARAMETERS # RNA BACKBONE PARAMETERS TUT
        ##################
        # Atom must be a tuple (atid, type, name, chargegrp, charge, mass)
        self.bb_atoms = [
                    (0, "Q1n", "BB1", 1, -1, 72), 
                    (1, "C6", "BB2", 1, 0, 60),
                    (2, "N2", "BB3", 1, 0, 60), 
                    ]
        # Bond must be a list of tuples [(connectivity), (parameters), (comment)]
        # Comments are very important: that's how we will indentify between bonds of the same type later
        self.bb_bonds = [
                    [(0, 1), (1,  0.350, 25000), ('BB1-BB2')],
                    [(1, 2), (1,  0.239, 25000), ('BB2-BB3')],          
                    [(1, 0), (1,  0.378, 12000), ('BB2-BB1n')],
                    [(2, 0), (1,  0.412, 12000), ('BB3-BB1n')]
                    ]
        self.bb_angles = [
                    [(0, 1, 0), (10,  110.0, 50), ('BB1-BB2-BB1n')],          
                    [(1, 0, 1), (10,  121.0, 180), ('BB2-BB1n-BB2n')],
                    [(0, 1, 2), (10,  143.0, 300), ('BB1-BB2-BB3')]
                    ]
        self.bb_dihs = [
                    [(0, 1, 0, 1), (1,    0.0, 25.0, 1), ('BB1-BB2-BB1n-BB2n')],          
                    [(-2, 0, 1, 0), (1,    0.0, 25.0, 1), ('BB2p-BB1-BB2-BB1n')],
                    [(-2, 0, 1, 2), (1, -112.0, 15.0, 1), ('BB2p-BB1-BB2-BB3n')]
                    ]
        self.bb_cons = []
        self.bb_excls = [
                    [(0, 2), (), ('BB1-BB3')], 
                    [(2, 0), (), ('BB3-BB1n')]
                    ]
        self.bb_pairs = []
        self.bb_vs3s = []
        # list with all bonded parameters
        self.bb_blist = [self.bb_bonds, self.bb_angles, self.bb_dihs, self.bb_cons, 
            self.bb_excls, self.bb_pairs, self.bb_vs3s]
       
        a_atoms = [
                (3, "TA0", "SC1", 2, 0, 45), 
                (4, "TA1", "SC2", 2, 0, 0), 
                (5, "TA2", "SC3", 2, 0, 45), 
                (6, "TA3", "SC4", 2, 0, 45), 
                (7, "TA4", "SC5", 2, 0, 0), 
                ]
        c_atoms = [
                (3, "TY0", "SC1", 2, 0, 37), 
                (4, "TY1", "SC2", 2, 0, 37), 
                (5, "TY2", "SC3", 2, 0, 0), 
                (6, "TY3", "SC4", 2, 0, 37), 
                ]
        g_atoms = [
                (3, "TG0", "SC1", 2, 0, 50), 
                (4, "TG1", "SC2", 2, 0, 0), 
                (5, "TG2", "SC3", 2, 0, 50), 
                (6, "TG3", "SC4", 2, 0, 0), 
                (7, "TG4", "SC5", 2, 0, 50), 
                (8, "TG5", "SC6", 2, 0, 0), 
                ]
        u_atoms = [
                (3, "TU0", "SC1", 2, 0, 37), 
                (4, "TU1", "SC2", 2, 0, 37), 
                (5, "TU2", "SC3", 2, 0, 0), 
                (6, "TU3", "SC4", 2, 0, 37), 
                ]
        sc_atoms = a_atoms, c_atoms, g_atoms, u_atoms
        self.mapdict = dict(zip(self.resnames, sc_atoms))

    def sc_atoms(self, resname):
        return self.mapdict[resname]     

  
class martini31nucleic(NucleicForceField):
    
    # FF mapping
    bb_mapping = nsplit("P OP1 OP2 O5' O3' O1P O2P", 
                        "C5' 1H5' 2H5' H5' H5'' C4' H4' O4' C3' H3'", 
                        "C1' C2' O2' O4'")     
    mapping = {
        "A":  bb_mapping + nsplit(
                        "C8",
                        "N3 C4",
                        "C2",
                        "N1",
                        "N6 C6 H61 H62",
                        "N7 C5", ),
        "C":  bb_mapping + nsplit(
                        "C6",
                        "O2",
                        "N3",
                        "N4 C4 H41 H42",
                        "C2"),
        "G":  bb_mapping + nsplit(
                        "C8",
                        "C4 N3",
                        "C2 N2 H22 H21",
                        "N1", 
                        "O6",
                        "C5 N7",
                        "H1",
                        "C6"),
        "U":  bb_mapping + nsplit(
                        "C6",
                        "O2",
                        "N3",
                        "O4",
                        "C2",
                        "H3",
                        "C4",),
    }    
    
    # NucleicForceField.update_non_standard_mapping(mapping)
    
    
    def __init__(self):
        
        # parameters are defined here for the following (protein) forcefields:
        self.name = 'martini31nucleic'
        
        # Charged types:
        charges = {"TDU":0.5,   "TA1":0.4, "TA2":-0.3, "TA3":0.5, "TA4":-0.8, "TA5":0.6, "TA6":-0.4, 
                                "TY1":0.0, "TY2":-0.5, "TY3":-0.6, "TY4":0.6, "TY5":0.5,
                                "TG1":0.3, "TG2":0.0, "TG3":0.3, "TG4":-0.3, "TG5":-0.5, "TG6":-0.6, "TG7":0.3, "TG8":0.5,
                                "TU1":0.0, "TU2":-0.5, "TU3":-0.5, "TU4":-0.5, "TU5":0.5, "TU6":0.5, "TU7":0.5,}  
        self.charges = {key: value * 1.8 for key, value in charges.items()}
        self.bbcharges = {"BB1":-1}                                                                                                      
        
        # Not all (eg Elnedyn) forcefields use backbone-backbone-sidechain angles and BBBB-dihedrals.
        self.UseBBSAngles          = False 
        self.UseBBBBDihedrals      = False
        
        ##################
        # DNA PARAMETERS #
        ##################

        # DNA BACKBONE PARAMETERS
        self.dna_bb = {
            'atom'  : spl("Q0 SN0 SC2"),
            'bond'  : [],         
            'angle' : [],           
            'dih'   : [],
            'excl'  : [],
            'pair'  : [],
        }
        # DNA BACKBONE CONNECTIVITY
        self.dna_con  = {
            'bond'  : [],
            'angle' : [],
            'dih'   : [],
            'excl'  : [],
            'pair'  : [],
        }

        self.bases = {}
        self.base_connectivity = {}
       

        ##################
        # RNA PARAMETERS # @ff
        ##################

        # RNA BACKBONE PARAMETERS TUT
        self.rna_bb = {
            'atom'  : spl("Q1 N4 N6"),    
            'bond'  : [(1,  0.349, 18000),          
                       (1,  0.377, 12000),
                       (1,  0.240, 18000),
                       (1,  0.412, 12000)],          
            'angle' : [(10,  119.0,  27),       
                       (10,  118.0, 140),
                       (10,  138.0, 180)],        
            'dih'   : [(3,    13,  -7, -25,  -6,  25, -2),  # (3,   10,  -8, 22, 8, -26, -6) # (3,   4,  -3, 9, 3, -10, -3)
                       (1,     0,   6,  1),
                       (1,   -112.0,   15,  1),],  # (1,     15.0,   5, 1)
            'excl'  : [(), (), ()],
            'pair'  : [],
        }
        # RNA BACKBONE CONNECTIVITY
        self.rna_con  = {
            'bond'  : [(0, 1),
                       (1, 0),
                       (1, 2),
                       (2, 0)],
            'angle' : [(0, 1, 0),
                       (1, 0, 1),
                       (0, 1, 2),],
            'dih'   : [(0, 1, 0, 1),
                       (1, 0, 1, 0),
                       (1, 0, 1, 2),],
            'excl'  : [(2, 0), (0, 2),],
            'pair'  : [],
        }
        
        a_itp, c_itp, g_itp, u_itp = NucleicForceField.read_itps(rna_system, 'polar', 'new')

        # ADENINE
        mapping = [spl("TA1 TA2 TA3 TA4 TA5 TA6")]
        connectivity, itp_params = martini31nucleic.itp_to_indata(a_itp)
        parameters = mapping + itp_params
        self.update_adenine(mapping, connectivity, itp_params)
        
        # CYTOSINE
        mapping = [spl("TY1 TY2 TY3 TY4 TY5")]
        connectivity, itp_params = martini31nucleic.itp_to_indata(c_itp)
        parameters = mapping + itp_params
        self.update_cytosine(mapping, connectivity, itp_params)
        
        # GUANINE
        mapping = [spl("TG1 TG2 TG3 TG4 TG5 TG6 TG7 TG8")]
        connectivity, itp_params = martini31nucleic.itp_to_indata(g_itp)
        parameters = mapping + itp_params
        self.update_guanine(mapping, connectivity, itp_params)
        
        # URACIL
        mapping = [spl("TU1 TU2 TU3 TU4 TU5 TU6 TU7")]
        connectivity, itp_params = martini31nucleic.itp_to_indata(u_itp)
        parameters = mapping + itp_params
        self.update_uracil(mapping, connectivity, itp_params)
        
        super().__init__()
        
 


    # def update_adenine(self, mapping, connectivity, itp_params):
    #     parameters = mapping + itp_params
    #     self.bases.update({"A": parameters})
    #     self.base_connectivity.update({"A": connectivity})
    #     self.bases.update({"RA3": parameters})
    #     self.base_connectivity.update({"RA3": connectivity})
    #     self.bases.update({"RA5": parameters})
    #     self.base_connectivity.update({"RA5": connectivity})
    #     self.bases.update({"2MA": parameters})
    #     self.base_connectivity.update({"2MA": connectivity})
    #     self.bases.update({"DMA": parameters})
    #     self.base_connectivity.update({"DMA": connectivity})
    #     self.bases.update({"SPA": parameters})
    #     self.base_connectivity.update({"SPA": connectivity})
    #     self.bases.update({"RAP": parameters})
    #     self.base_connectivity.update({"RAP": connectivity})
    #     self.bases.update({"6MA": parameters})
    #     self.base_connectivity.update({"6MA": connectivity})
        
    # def update_cytosine(self, mapping, connectivity, itp_params):  
    #     parameters = mapping + itp_params
    #     self.bases.update({"C": parameters})
    #     self.base_connectivity.update({"C": connectivity})
    #     self.bases.update({"RC3": parameters})
    #     self.base_connectivity.update({"RC3": connectivity})
    #     self.bases.update({"RC5": parameters})
    #     self.base_connectivity.update({"RC5": connectivity})
    #     self.bases.update({"MRC": parameters})
    #     self.base_connectivity.update({"MRC": connectivity})
    #     self.bases.update({"5MC": parameters})
    #     self.base_connectivity.update({"5MC": connectivity})
    #     self.bases.update({"NMC": parameters})
    #     self.base_connectivity.update({"NMC": connectivity})    
   
    # def update_guanine(self, mapping, connectivity, itp_params):
    #     parameters = mapping + itp_params
    #     self.bases.update({"G": parameters})
    #     self.base_connectivity.update({"G": connectivity})
    #     self.bases.update({"RG3": parameters})
    #     self.base_connectivity.update({"RG3": connectivity})
    #     self.bases.update({"RG5": parameters})
    #     self.base_connectivity.update({"RG5": connectivity})
    #     self.bases.update({"MRG": parameters})
    #     self.base_connectivity.update({"MRG": connectivity})
    #     self.bases.update({"1MG": parameters})
    #     self.base_connectivity.update({"1MG": connectivity})
    #     self.bases.update({"2MG": parameters})
    #     self.base_connectivity.update({"2MG": connectivity})
    #     self.bases.update({"7MG": parameters})
    #     self.base_connectivity.update({"7MG": connectivity})

    # def update_uracil(self, mapping, connectivity, itp_params):
    #     parameters = mapping + itp_params
    #     self.bases.update({"U": parameters})
    #     self.base_connectivity.update({"U": connectivity})
    #     self.bases.update({"RU3": parameters})
    #     self.base_connectivity.update({"RU3": connectivity})
    #     self.bases.update({"RU5": parameters})
    #     self.base_connectivity.update({"RU5": connectivity})
    #     self.bases.update({"MRU": parameters})
    #     self.base_connectivity.update({"MRU": connectivity})
    #     self.bases.update({"DHU": parameters})
    #     self.base_connectivity.update({"DHU": connectivity})        
    #     self.bases.update({"PSU": parameters})
    #     self.base_connectivity.update({"PSU": connectivity})
    #     self.bases.update({"3MP": parameters})
    #     self.base_connectivity.update({"3MP": connectivity})
    #     self.bases.update({"3MU": parameters})
    #     self.base_connectivity.update({"3MU": connectivity})
    #     self.bases.update({"4SU": parameters})
    #     self.base_connectivity.update({"4SU": connectivity})        
    #     self.bases.update({"5MU": parameters})
    #     self.base_connectivity.update({"5MU": connectivity})
     
    # @staticmethod
    # def update_non_standard_mapping(mapping):
    #     mapping.update({"RA3":mapping["A"],
    #                     "RA5":mapping["A"],
    #                     "2MA":mapping["A"],
    #                     "6MA":mapping["A"],
    #                     "RAP":mapping["A"],
    #                     "DMA":mapping["A"],
    #                     "DHA":mapping["A"],
    #                     "SPA":mapping["A"],
    #                     "RC3":mapping["C"],
    #                     "RC5":mapping["C"],
    #                     "5MC":mapping["C"],
    #                     "3MP":mapping["C"],
    #                     "MRC":mapping["C"],
    #                     "NMC":mapping["C"],
    #                     "RG3":mapping["G"],
    #                     "RG5":mapping["G"],
    #                     "1MG":mapping["G"],
    #                     "2MG":mapping["G"],
    #                     "7MG":mapping["G"],
    #                     "MRG":mapping["G"],
    #                     "RU3":mapping["U"],
    #                     "RU5":mapping["U"],
    #                     "4SU":mapping["U"], 
    #                     "DHU":mapping["U"], 
    #                     "PSU":mapping["U"],
    #                     "5MU":mapping["U"],
    #                     "3MU":mapping["U"],
    #                     "3MP":mapping["U"],
    #                     "MRU":mapping["U"],
    #     })
