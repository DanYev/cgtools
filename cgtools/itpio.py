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

    with open(output_file, 'w') as file:
        file.write('[ atomtypes ]' + '\n')
        for key in dict_of_names.keys():
            file.write(f"{key}  45.000  0.000  A  0.0  0.0\n")
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
                        parts[3] = "3.000000e-01"
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
    dict_of_names = {   'TA0': 'TN3', 
                        'TA1': 'TN3a',
                        'TA2': 'TN5a', 
                        'TA3': 'TP1d',
                        'TA4': 'TN1a', 
                        'TY0': 'SN3',
                        'TY1': 'TP3a',
                        'TY2': 'TP1a',
                        'TY3': 'TP3d',
                        'TG0': 'TN3', 
                        'TG1': 'TN3a',
                        'TG2': 'TP1d', 
                        'TG3': 'TP1d',
                        'TG4': 'TP2a', 
                        'TG5': 'TN1a',
                        'TU0': 'SN3',
                        'TU1': 'TP2a',
                        'TU2': 'TN5d',
                        'TU3': 'TP1a',
    }
    dict_of_names = {   'TA0': 'TN3', 
                        'TA1': 'TN3',
                        'TA2': 'TN5', 
                        'TA3': 'TP1',
                        'TA4': 'TN1', 
                        'TY0': 'SN3',
                        'TY1': 'TP3',
                        'TY2': 'TP1',
                        'TY3': 'TP3',
                        'TG0': 'TN3', 
                        'TG1': 'TN3',
                        'TG2': 'TP1', 
                        'TG3': 'TP1',
                        'TG4': 'TP2', 
                        'TG5': 'TN1',
                        'TU0': 'SN3',
                        'TU1': 'TP2',
                        'TU2': 'TN5',
                        'TU3': 'TP1',
    }
    out_file = 'cgtools/itp/martini_RNA.itp'
    
    make_in_terms('cgtools/itp/martini.itp', out_file, dict_of_names)
    for new_name, old_name in dict_of_names.items():
        make_cross_terms('cgtools/itp/martini.itp', out_file, old_name, new_name)
        
        
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
