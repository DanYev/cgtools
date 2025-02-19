#!/usr/bin/env python3
"""
Gromacs Topology Formatter
"""

def bond2str(atoms=None, parameters=None, comment=''):
    """
    Returns a formatted string for a bond entry in a Gromacs ITP file.
    
    Parameters:
        atoms (list of int): The two atom indices involved in the bond.
        parameters (list of float): Bond parameters (e.g., bond length, force constant).
        comment (str): An optional comment to append.
        
    Returns:
        str: A formatted bond entry string.
        
    Example:
        >>> print(bond2str(atoms=[1, 2], parameters=[1 0.153, 345.0], comment="Harmonic bond"))
             1     2     1   0.1530  345.0000 ; Harmonic bond
    """   
    # Combine everything. 
    result = "   ".join(str(atom) for atom in atoms) + "   " + "   ".join(str(param) for param in parameters) 
    if comment: # Append comment if provided.
        result += " ; " + comment
    return result


# Example usage:
if __name__ == "__main__":
    bond_line = bond2str(atoms=[1, 2, 3], parameters=[2, 180, 345.0], comment="Angle")
    print(bond_line)


