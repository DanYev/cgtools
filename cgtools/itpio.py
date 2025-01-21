import shutil as sh

def read_itp(filename):
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
                itp_data[tag] = {}
            else:
                connectivity, parameters = split_conn_params(line, tag)
                key = connectivity
                value = parameters
                itp_data[tag][key] = value
    return itp_data
    
    
def write_itp(filename, itp_data):
    with open(filename, 'w') as file:
        for section, data_dict in itp_data.items():
            file.write(f'[ {section} ]\n')
            for key, value in data_dict.items():
                l1 = [str(i) for i in key]
                l2 = [str(i) for i in value]
                str1 = ' '.join(l1)
                str2 = ' '.join(l2)
                file.write(f'{str1} {str2}\n')
                
                
def split_conn_params(line, tag):
    data = line.split()
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
    if parameters and parameters[-1].startswith(';'):
        parameters.pop(-1)
    if parameters:
        parameters[0] = int(parameters[0])
        parameters[1:] = [float(i) for i in parameters[1:]]
    connectivity  = tuple([int(i) for i in connectivity])
    parameters = tuple(parameters)
    return connectivity, parameters
    
    
def make_in_terms(input_file, output_file, dict_of_names):
    tag = None
    pairs = []
    
    def get_sigma(b1, b2):
        list_of_pairs_1 = { 
            ('TA3', 'TU2'),('TA4', 'TU3'), ('TA5', 'TU4'), 
            ('TG3', 'TY2'), ('TG4', 'TY3'), ('TG5', 'TY4'),
        }
        list_of_pairs_2 = {
            ('TA3', 'TU3'), ('TA4', 'TU2'), ('TA4', 'TU4'), ('TA5', 'TU3'), 
            ('TG3', 'TY3'), ('TG4', 'TY2'), ('TG4', 'TY4'), ('TG5', 'TY3'),
        }
        if (b1, b2) in list_of_pairs_1 or (b2, b1) in list_of_pairs_1:
            sigma = "2.65000e-01"
        elif (b1, b2) in list_of_pairs_2 or (b2, b1) in list_of_pairs_2:
             sigma = "2.750000e-01"
        else:
            sigma = "3.3000000e-01"
        return sigma

    with open(output_file, 'w') as file:
        file.write('[ atomtypes ]' + '\n')
        dict_of_vdw = {   
            'TA1': ('3.200000e-01', '1.368000e-01'), 
            'TA2': ('3.200000e-01', '1.368000e-01'),
            'TA3': ('3.200000e-01', '1.368000e-01'), 
            'TA4': ('2.800000e-01', '1.368000e-01'),
            'TA5': ('2.800000e-01', '1.368000e-01'),
            'TA6': ('3.200000e-01', '1.368000e-01'), 
            'TY1': ('3.200000e-01', '1.368000e-01'),
            'TY2': ('3.200000e-01', '1.368000e-01'),
            'TY3': ('3.200000e-01', '1.368000e-01'),
            'TY4': ('3.200000e-01', '1.368000e-01'),
            'TY5': ('3.200000e-01', '1.368000e-01'),
            'TG1': ('3.200000e-01', '1.368000e-01'), 
            'TG2': ('3.200000e-01', '1.368000e-01'),
            'TG3': ('2.800000e-01', '1.368000e-01'), 
            'TG4': ('2.800000e-01', '1.368000e-01'),
            'TG5': ('2.800000e-01', '1.368000e-01'), 
            'TG6': ('3.200000e-01', '1.368000e-01'),
            'TG7': ('0.400000e-01', '1.368000e-01'),
            'TG8': ('3.200000e-01', '1.368000e-01'),
            'TU1': ('3.200000e-01', '1.368000e-01'),
            'TU2': ('3.200000e-01', '1.368000e-01'),
            'TU3': ('2.800000e-01', '1.368000e-01'),
            'TU4': ('3.200000e-01', '1.368000e-01'),
            'TU5': ('3.200000e-01', '1.368000e-01'),
            'TU6': ('0.400000e-01', '1.368000e-01'),
            'TU7': ('3.200000e-01', '1.368000e-01'),
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
                        'TA3': 'TN1d', 
                        'TA4': 'TN6ar',
                        'TA5': 'TN6dr',
                        'TA6': 'TN1a', 
                        'TY1': 'SC3',
                        'TY2': 'TN5a',
                        'TY3': 'TN6a',
                        'TY4': 'TN3d',
                        'TY5':  None,
                        'TG1': 'SC5', 
                        'TG2': 'TN1a',
                        'TG3': 'TN3dr', 
                        'TG4': 'TN6dr',
                        'TG5': 'TN6ar', 
                        'TG6': 'TN1a',
                        'TG7':  None,
                        'TG8':  None,
                        'TU1': 'SC3',
                        'TU2': 'TN4a',
                        'TU3': 'TN6d',
                        'TU4': 'TN4a',
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
