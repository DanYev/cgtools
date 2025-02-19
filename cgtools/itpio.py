import shutil as sh
from typing import List, Tuple, Any

###################################
## Generic functions ## 
###################################

def read_itp(filename):
    """
    Reads a Gromacs ITP file and organizes its contents by section.

    This function parses the ITP file specified by `filename`, splitting it into sections
    (such as 'bonds', 'angles', etc.). For each section, it creates a list of entries,
    where each entry is a list containing a tuple of connectivity indices, a tuple of parameters,
    and an optional comment string.

    Args:
        filename (str): The path to the ITP file.

    Returns:
        dict: A dictionary where the keys are section names (str) and the values are lists of
              entries. Each entry is of the form: [connectivity (tuple), parameters (tuple), comment (str)].

    Example:
        >>> itp_data = read_itp("topology.itp")
        >>> print(itp_data.get("bonds"))
    """
    itp_data = {}
    current_section = None
    with open(filename, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            if line.strip() == '' or line.strip().startswith(';'):
                continue
            # Detect section headers
            if line.startswith('[') and line.endswith(']\n'):
                tag = line.strip()[2:-2]
                itp_data[tag] = []
            else:
                connectivity, parameters, comment = line2bond(line, tag)
                itp_data[tag].append([connectivity, parameters, comment])
    return itp_data


def line2bond(line, tag):
    """
    Parses a line from an ITP file and returns connectivity, parameters, and comment based on the section.

    This function splits the line at the first semicolon to separate the data from the comment.
    It then splits the data into tokens and, based on the section specified by `tag`, extracts
    connectivity indices and parameters. The function converts connectivity to a tuple of ints,
    and parameters to a tuple of numeric values (with the first parameter as an int and the rest as floats).

    Args:
        line (str): A line from the ITP file.
        tag (str): The section tag (e.g., 'bonds', 'angles', etc.) determining the parsing rules.

    Returns:
        tuple:
            connectivity (tuple of int): The connectivity indices.
            parameters (tuple): The associated parameters (numeric values).
            comment (str): The comment string, if present (empty string otherwise).

    Example:
        >>> conn, params, comm = line2bond(" 1  2  1 0.153 345.0 ; Harmonic bond", "bonds")
    """
    data, sep, comment = line.partition(';')
    data = data.split()
    comment = comment.strip()
    if tag == 'bonds':
        connectivity = data[:2]
        parameters = data[2:]
    elif tag == 'constraints':
        connectivity = data[:2]
        parameters = data[2:]
        parameters.append(None)
    elif tag == 'angles':
        connectivity = data[:3]
        parameters = data[3:]
    elif tag == 'dihedrals':
        connectivity = data[:4]
        parameters = data[4:]
    elif tag == 'virtual_sites3':
        connectivity = data[:4]
        parameters = data[4:]
    else:
        connectivity = data
        parameters = []
    if parameters:
        parameters[0] = int(parameters[0])
        parameters[1:] = [float(i) for i in parameters[1:]]
    connectivity = tuple([int(i) for i in connectivity])
    parameters = tuple(parameters)
    return connectivity, parameters, comment


def bond2line(connectivity=None, parameters='', comment=''):
    """
    Returns a formatted string for a bond entry in a Gromacs ITP file.

    The function formats the given atom indices and bond parameters into a single line.
    Atom indices and parameters are separated by a consistent amount of whitespace. An
    optional comment can be appended, preceded by a semicolon.

    Args:
        atoms (list of int): The atom indices involved in the bond.
        parameters (list of float): Bond parameters (e.g., bond length, force constant).
        comment (str): An optional comment to append at the end of the line.

    Returns:
        str: A formatted bond entry string.

    Example:
        >>> print(bond2line(atoms=[1, 2], parameters=[1, 0.153, 345.0], comment="Harmonic bond"))
             1     2     1   0.153   345.0 ; Harmonic bond
    """
    # Format each connectivity value as a 5-character wide integer.
    connectivity_str = "   ".join(f"{int(atom):5d}" for atom in connectivity)
    type_str = ''
    parameters_str = ''
    if parameters:
        # Format each type as a 2-character wide integer.
        type_str = "   ".join(f"{int(parameters[0]):2d}")
        # Format each parameter as a 7-character wide float with 4 decimal places.
        parameters_str = "   ".join(f"{float(param):7.4f}" for param in parameters[1:])
    line = connectivity_str + "   " + type_str + "   " + parameters_str
    if comment:  # Append comment if provided.
        line += " ; " + comment
    line += "\n"
    return line


def format_header(molecule_name='molecule', forcefield='', version='', arguments='') -> List[str]:
    """
    Formats the header of the topology file.
    """
    lines = [f"; MARTINI ({forcefield}) Coarse Grained topology file for \"{molecule_name}\"\n"]
    lines.append(f"; Created by version {version} \n; Using the following options: {arguments}\n")
    lines.append("; " + "#" * 100 + "\n")
    # lines.append("; " + "#" * 100 + "\n")
    return lines


def format_moleculetype_section(molecule_name='molecule', nrexcl=1) -> List[str]:
    """
    Formats the moleculetype section.
    """
    lines = ["\n[ moleculetype ]\n"]
    lines.append("; Name         Exclusions\n")
    lines.append(f"{molecule_name:<15s} {nrexcl:3d}\n")
    return lines


def format_atoms_section(atoms: List[Tuple]) -> List[str]:
    """
    Formats the atoms section for a Gromacs ITP file.
    Args:
        atoms (List[Tuple[Any, ...]]): A list of atom records, where each record is a tuple.
                                       The tuple should have 8 or 9 elements depending on the atom.
        atom is the tuple: (atid, type, resid, resname, name, chargegrp, charge, mass, comment)
    Returns:
        List[str]: A list of formatted strings representing the atoms section.
    """
    lines = ["\n[ atoms ]\n"]
    fs8 = '%5d %5s %5d %5s %5s %5d %7.4f ; %s'
    fs9 = '%5d %5s %5d %5s %5s %5d %7.4f %7.4f ; %s'
    for atom in atoms:
        # Choose format based on length of the atom tuple.
        line = fs9 % atom if len(atom) == 9 else fs8 % atom
        line += "\n"
        lines.append(line) 
    return lines


def format_bonded_section(header: str, bonds: List[List]) -> List[str]:
    """
    Formats the atoms section for a Gromacs ITP file.
    Args:
        bonds (List[Tuple[Any, ...]]): A list of atom records, where each record is a tuple.
                                       The tuple should have 8 or 9 elements depending on the atom.
    Returns:
        List[str]: A list of formatted strings representing the atoms section.
    """
    lines = [f"\n[ {header} ]\n"]
    for bond in bonds:
        line = bond2line(*bond)
        lines.append(line) 
    return lines


def format_posres_section(atoms: List[Tuple], posres_fc=1000, selection=['BB1', 'BB3', 'SC1']) -> List[str]:
    """
    Formats the position restraints section.
    atom is the tuple: (atid, type, resid, resname, name, chargegrp, charge, mass, comment)
    """
    lines = ["\n#ifdef POSRES\n",
                "#define POSRES_FC %.2f\n" % posres_fc,
                " [ position_restraints ]\n"]
    for atom in atoms:
        if atom[4] in selection:
            lines.append('  %5d    1    POSRES_FC    POSRES_FC    POSRES_FC\n' % atom[0])
    lines.append("#endif")
    return lines


def write_itp(filename, lines):
    with open(filename, 'w') as file:
        for line in lines:
            file.write(line)


###################################
## HL functions for martini_rna ## 
###################################


def make_in_terms(input_file, output_file, dict_of_names):
    tag = None
    pairs = []
    
    def get_sigma(b1, b2):
        list_of_pairs_1 = { 
            ('TA4', 'TU3'), ('TA5', 'TU4'), 
            ('TG3', 'TY2'), ('TG4', 'TY3'), ('TG5', 'TY4'),
        }
        list_of_pairs_2 = {
            ('TA4', 'TU2'), ('TA4', 'TU4'), ('TA5', 'TU3'), 
            ('TG3', 'TY3'), ('TG4', 'TY2'), ('TG4', 'TY4'), ('TG5', 'TY3'),
        }
        list_of_pairs_3 = {
            ('TA4', 'TY3'), ('TA5', 'TY4'), 
            ('TA4', 'TY2'), ('TA4', 'TY4'), ('TA5', 'TY3'), 
            ('TG3', 'TU2'), ('TG4', 'TU3'), ('TG5', 'TU4'),
            ('TG3', 'TU3'), ('TG4', 'TU2'), ('TG4', 'TU4'), ('TG5', 'TU3'),
        }
        if (b1, b2) in list_of_pairs_1 or (b2, b1) in list_of_pairs_1:
            sigma = "2.75000e-01"
        elif (b1, b2) in list_of_pairs_2 or (b2, b1) in list_of_pairs_2:
             sigma = "2.750000e-01"
        elif (b1, b2) in list_of_pairs_3 or (b2, b1) in list_of_pairs_3:
             sigma = "2.750000e-01"
        else:
            sigma = "3.300000e-01"
        return sigma

    with open(output_file, 'w') as file:
        file.write('[ atomtypes ]' + '\n')
        dict_of_vdw = {   
            'TA1': ('3.250000e-01', '1.000000e-01'), 
            'TA2': ('3.250000e-01', '1.000000e-01'),
            'TA3': ('3.250000e-01', '1.000000e-01'), 
            'TA4': ('2.800000e-01', '1.368000e-01'),
            'TA5': ('2.800000e-01', '1.368000e-01'),
            'TA6': ('3.250000e-01', '1.000000e-01'), 
            'TY1': ('3.250000e-01', '1.000000e-01'),
            'TY2': ('2.800000e-01', '1.000000e-01'),
            'TY3': ('2.800000e-01', '1.000000e-01'),
            'TY4': ('2.800000e-01', '1.000000e-01'),
            'TY5': ('3.250000e-01', '1.000000e-01'),
            'TG1': ('3.250000e-01', '1.000000e-01'), 
            'TG2': ('3.250000e-01', '1.000000e-01'),
            'TG3': ('2.800000e-01', '1.368000e-01'), 
            'TG4': ('2.800000e-01', '1.368000e-01'),
            'TG5': ('2.800000e-01', '1.368000e-01'), 
            'TG6': ('3.250000e-01', '1.000000e-01'),
            'TG7': ('0.400000e-01', '1.368000e-01'),
            'TG8': ('3.250000e-01', '1.000000e-01'),
            'TU1': ('3.250000e-01', '1.000000e-01'),
            'TU2': ('2.800000e-01', '1.000000e-01'),
            'TU3': ('2.800000e-01', '1.368000e-01'),
            'TU4': ('2.800000e-01', '1.000000e-01'),
            'TU5': ('3.250000e-01', '1.000000e-01'),
            'TU6': ('0.400000e-01', '1.368000e-01'),
            'TU7': ('3.250000e-01', '1.000000e-01'),
        }
        for key in dict_of_names.keys():
            file.write(f"{key}  45.000  0.000  A  {dict_of_vdw[key][0]}  {dict_of_vdw[key][1]}\n")
        file.write('\n' + '[ nonbond_params ]' + '\n')
                    
    with open(input_file, 'r') as file:
        lines = file.readlines()
    with open(output_file, 'a') as file:
        for line in lines:
            if line.startswith(';') or len(line.split()) < 2:
                continue
            parts = line.split()
            atom_name_1 = parts[0].strip()
            atom_name_2 = parts[1].strip()
            if atom_name_1 in dict_of_names.values() and atom_name_2 in dict_of_names.values():
                keys_1 = [key for key, value in dict_of_names.items() if value == atom_name_1]
                keys_2 = [key for key, value in dict_of_names.items() if value == atom_name_2]
                for key_1 in keys_1:
                    for key_2 in keys_2:
                        if (key_1, key_2) in pairs or (key_2, key_1) in pairs:
                            continue
                        pairs.append((key_1, key_2))
                        parts[0] = key_1
                        parts[1] = key_2
                        parts[3] = get_sigma(key_1, key_2)
                        file.write('  '.join(parts) + '\n')
            
            
def make_cross_terms(input_file, output_file, old_name, new_name):
    switch = False
    with open(input_file, 'r') as file:
        lines = file.readlines()
    with open(output_file, 'a') as file:
        for line in lines:
            if line.startswith(';') or len(line.split()) < 2:
                continue
            if line.startswith('[ nonbond_params ]'):
                switch = True
            if switch:
                parts = line.split()
                atom_name_1 = parts[0].strip()
                atom_name_2 = parts[1].strip()
                if atom_name_1 == old_name:
                    parts[0] = new_name
                    file.write('  '.join(parts) + '\n')
                    continue
                if atom_name_2 == old_name:
                    parts[1] = new_name
                    file.write('  '.join(parts) + '\n')
                    continue

            else:
                continue

def make_marnatini_itp():
    # data = read_itp('../itp/working/1RNA_A.itp')
    # write_itp('test.itp', data)
    # print(data)
    dict_of_names = {   'TA0': 'TN1', 
                        'TA1': 'TN3a',
                        'TA2': 'TP1a', 
                        'TA3': 'TP1d',
                        'TA4': 'TP1a', 
                        'TY0': 'SN2',
                        'TY1': 'TP3a',
                        'TY2': 'TP1a',
                        'TY3': 'TP3d',
                        'TG0': 'TP1', 
                        'TG1': 'TN3a',
                        'TG2': 'TP1d', 
                        'TG3': 'TP1d',
                        'TG4': 'TP2a', 
                        'TG5': 'TP1a',
                        'TU0': 'SN2',
                        'TU1': 'TP2a',
                        'TU2': 'TP1d',
                        'TU3': 'TP1a',
    }
    dict_of_names = {   'TA1': 'SC5', 
                        'TA2': 'TN1a',
                        'TA3': 'TC6', 
                        'TA4': 'TN4r',
                        'TA5': 'TN4r',
                        'TA6': 'TN1a', 
                        'TY1': 'SC3',
                        'TY2': 'TN3',
                        'TY3': 'TN4',
                        'TY4': 'TN3',
                        'TY5':  None,
                        'TG1': 'SC5', 
                        'TG2': 'TN1a',
                        'TG3': 'TN3r', 
                        'TG4': 'TN4r',
                        'TG5': 'TN4r', 
                        'TG6': 'TN1a',
                        'TG7':  None,
                        'TG8':  None,
                        'TU1': 'SC3',
                        'TU2': 'TN3',
                        'TU3': 'TN4',
                        'TU4': 'TN3',
                        'TU5':  None,
                        'TU6':  None,
                        'TU7':  None,
    }
    out_file = 'cgtools/itp/martini_RNA.itp'
    
    make_in_terms('cgtools/itp/martini.itp', out_file, dict_of_names)
    for new_name, old_name in dict_of_names.items():
        make_cross_terms('cgtools/itp/martini.itp', out_file, old_name, new_name)
    sh.copy(out_file, '/scratch/dyangali/cgtools/systems/dsRNA/topol/martini_v3.0.0_rna.itp')
    sh.copy(out_file, '/scratch/dyangali/cgtools/systems/ssRNA/topol/martini_v3.0.0_rna.itp')
    sh.copy(out_file, '/scratch/dyangali/maRNAtini_sims/dimerization_pmf_us/topol/martini_RNA.itp')
    sh.copy(out_file, '/scratch/dyangali/maRNAtini_sims/angled_dimerization_pmf_us/topol/martini_RNA.itp') # angled_dimerization_pmf_us
        
def make_ions_itp():
    import pandas as pd
    dict_of_names = {   'TMG': 'TD',
    }
    out_file = 'cgtools/itp/ions.itp'
    for new_name, old_name in dict_of_names.items():
        make_cross_terms('cgtools/itp/martini_v3.0.0.itp', out_file, old_name, new_name)
    df = pd.read_csv(out_file, sep='\\s+', header=None)
    df[3] -= 0.08
    tmp_file = 'cgtools/itp/ions_tmp.itp'
    df.to_csv(tmp_file, sep=' ', header=None, index=False, float_format='%.6e')
    
    out_file = 'cgtools/itp/martini_ions.itp'
    new_lines = ["[ atomtypes ]\n", 
        "TMG  45.000  0.000  A  0.0  0.0\n\n", 
        "[ nonbond_params ]\n", 
        "TMG TMG 1 3.580000e-01 1.100000e+00\n"]
    with open(tmp_file, "r") as file:
        original_content = file.readlines()
    with open(out_file, "w+") as file:
        file.writelines(new_lines + original_content)
        
        


if __name__ == '__main__':
    # make_ions_itp()
    make_marnatini_itp()
