import os
import sys
import shutil
import fileinput
import subprocess as sp

system = sys.argv[1]

def main():
    generate_topology_file()
    generate_gro_file()
    generate_pdb_file()


def generate_pdb_file():
    cgpdb_folder = f"systems/{system}/cgpdb"
    cgpdb_files = sorted(os.listdir(cgpdb_folder))
    with open(f"systems/{system}/system.pdb", 'w') as out_f:
        for file in cgpdb_files:
            if file.endswith(".pdb"):
                filepath = os.path.join(cgpdb_folder, file)
                with open(filepath, 'r') as in_f:
                    lines = in_f.readlines()
                    for line in lines:
                        if line.startswith("ATOM"):
                            out_f.write(line)


if __name__ == "__main__":
    main()


        
            
    