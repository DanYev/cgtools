import os

def read_b_factors(file_path):
    """
    Reads the B-factor file and returns a dictionary mapping residue numbers to B-factors.
    """
    b_factors = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) != 2:
                continue
            residue_number = int(parts[0])
            b_factor = float(parts[1])
            b_factors[residue_number] = b_factor
    return b_factors

def update_bfactors(pdb_file, b_factors, output_file):
    """
    Reads the PDB file, updates B-factors for each residue, and writes the updated PDB file.
    """
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    lines = [line for line in lines if line.startswith("ATOM")]
    updated_lines = []
    for idx, line in enumerate(lines):
        residue_number = int(line[22:26].strip())
        b_factor = b_factors[idx]
        updated_line = f"{line[:60]}{b_factor:6.2f}{line[66:]}"
        updated_lines.append(updated_line)
    with open(output_file, 'w') as file:
        file.writelines(updated_lines)

def main():
    pdb_file = "input.pdb"       # Replace with the path to your PDB file
    b_factor_file = "b_factors.txt"  # Replace with the path to your B-factor file
    output_file = "output.pdb"   # Path to save the updated PDB file
    
    if not os.path.exists(pdb_file):
        print(f"PDB file '{pdb_file}' does not exist.")
        return
    if not os.path.exists(b_factor_file):
        print(f"B-factor file '{b_factor_file}' does not exist.")
        return
    
    b_factors = read_b_factors(b_factor_file)
    update_pdb_b_factors(pdb_file, b_factors, output_file)
    print(f"Updated PDB file saved as '{output_file}'.")

if __name__ == "__main__":
    main()


    
    

  
