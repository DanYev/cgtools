           
def count_itp_atoms(file_path):
    in_atoms_section = False
    atom_count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip whitespace and check if it's a comment or empty line
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                # Detect the start of the [ atoms ] section
                if line.startswith("[ atoms ]"):
                    in_atoms_section = True
                    continue
                # Detect the start of a new section
                if in_atoms_section and line.startswith('['):
                    break
                # Count valid lines in the [ atoms ] section
                if in_atoms_section:
                    atom_count += 1
        return atom_count
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0