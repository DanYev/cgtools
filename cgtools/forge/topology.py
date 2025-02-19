#!/usr/bin/env python3
"""
Refactored Topology module.

This module contains the Topology class with improved readability,
modularity, and error checking. Many long methods have been split into
helper functions to improve maintainability.
"""

import logging
import numpy as np
from cgtools import itpio
from cgtools.forge.forcefields import NucleicForceField
from cgtools.forge.geometry import get_distance
from typing import Any, Dict, List, Optional, Tuple, Union


class Topology:
    def __init__(self, forcefield, sequence: List = [], secstruct: List = [], molname='molecule') -> None:
        """
        Initialize a Topology instance. Main attributes are: 
        1. atom - [atid, type, resid, resname, name, chargegrp, charge, mass, comment]
        2. bond - [connectivity, parameters, comment]

        forcefield - an instance of NucleicForceField Class
        :param other: Another Topology or Chain instance to initialize from.
        :param options: Dictionary of options, e.g. ForceField, Version, etc.
        :param name: Optional name for the topology.
        """
        self.ff = forcefield
        self.sequence = sequence
        self.name = molname
        self.nrexcl: int = 1
        self.atoms: List = []
        self.bonds: List = []
        self.angles: List = []
        self.dihs: List = []
        self.cons: List = []
        self.excls: List = []   
        self.pairs: List = []
        self.vs3s: List = []      
        self.posres: List = []
        self.elnet: List = []
        self.mapping: List = []
        self.natoms = len(self.atoms)
        # list with all bonded parameters as in the ff
        self.blist = [self.bonds, self.angles, self.dihs, self.cons, 
            self.excls, self.pairs, self.vs3s]  
        # Secondary structure     
        if secstruct:
            self.secstruct = secstruct
        else:
            self.secstruct = ['F' for item in self.sequence]

    def __iadd__(self, other) -> "Topology":
        """
        Implements in-place addition of another Topology.
        1. atom - (atid, type, resid, resname, name, chargegrp, charge, mass, comment)
        2. bond - [connectivity, parameters, comment]      
        :param other: Another Topology instance or object convertible to one.
        :return: self after merging the two topologies.
        """
        def update_atom(atom, atom_shift, residue_shift):
            atom[0] += atom_shift # Update atom numbers
            atom[2] += residue_shift # Update residue numbers
            atom[5] += atom_shift # Update charge group numbers
            return atom
        def update_bond(bond, atom_shift):
            conn = bond[0]
            conn = [idx + atom_shift for idx in conn]
            return [conn, bond[1], bond[2]]
        atom_shift = self.natoms
        residue_shift = len(self.sequence)
        last_atom = self.atoms[-1]
        # Update atoms
        new_atoms = other.atoms
        new_atoms = [update_atom(atom, atom_shift, residue_shift) for atom in new_atoms]   
        self.atoms.extend(new_atoms)
        # Update bonds
        for self_attrib, other_attrib in zip(self.blist, other.blist):
            other_attrib = [update_bond(bond, atom_shift) for bond in other_attrib]
            self_attrib.extend(other_attrib) 
        return self

    def __add__(self, other) -> "Topology":
        """
        Implements addition of two Topology objects.
        """
        # Create a copy of self. Assumes that the constructor can create a copy from self.
        new_top = self
        new_top += other  # Use __iadd__ to add the other topology
        return new_top

    def lines(self) -> list:
        """
        Returns the topology file representation as a list of lines.
        Returns:
            list: A list of strings, where each string is a line in the topology file.
        """
        lines = itpio.format_header(molname=self.name, forcefield=self.ff.name, 
                                    version='', arguments='')
        lines += itpio.format_sequence_section(self.sequence, self.secstruct)
        lines += itpio.format_moleculetype_section(molname=self.name, nrexcl=1)
        lines += itpio.format_atoms_section(self.atoms)
        lines += itpio.format_bonded_section('bonds', self.bonds)
        lines += itpio.format_bonded_section('angles', self.angles)
        lines += itpio.format_bonded_section('dihedrals', self.dihs)
        lines += itpio.format_bonded_section('constraints', self.cons)
        lines += itpio.format_bonded_section('exclusions', self.excls)
        lines += itpio.format_bonded_section('pairs', self.pairs)
        lines += itpio.format_bonded_section('virtual_sites3', self.vs3s)
        lines += itpio.format_bonded_section('bonds', self.elnet)
        lines += itpio.format_posres_section(self.atoms)
        logging.info('Created coarsegrained topology')
        return lines

    # def __str__(self) -> str:
    #     return "".join(self.lines())

    def write_itp(self, filename):
        with open(filename, 'w') as file:
            for line in self.lines():
                file.write(line)    


    @staticmethod
    def _update_bb_connectivity(conn, atid, reslen, prevreslen=None):
        """Update backbone connectivity indices for a residue.

        This method updates backbone connectivity indices provided by the force field (FF) for the current
        residue. It adjusts atom indices based on the length of the current residue, and for some
        dihedral definitions, uses the length of the previous residue.

        Args:
            conn (list of int): Connectivity indices from the force field.
                Negative indices indicate a connection relative to the previous residue.
            atid (int): Atom ID of the first atom in the current residue.
            reslen (int): Number of atoms in the current residue.
            prevreslen (Optional[int]): Number of atoms in the previous residue.
                If None, connectivity for negative indices is not updated and the original
                connectivity list is returned.

        Returns:
            List: A list of updated connectivity indices.

        Example:
            >>> conn = [0, 1, -1]
            >>> Topology._update_bb_connectivity(conn, 10, 5, prevreslen=4)
            (10, 11, 10 - 4 - 1 + 3)  # Adjusted accordingly.
        """
        result = []
        prev = -1
        for idx in conn:
            if idx < 0:
                if prevreslen:
                    result.append(atid - prevreslen + idx + 3) 
                    continue
                else:
                    return tuple(conn)
            if idx > prev:
                result.append(atid + idx)
            else:
                result.append(atid + idx + reslen)
                atid += reslen
            prev = idx
        return result

    @staticmethod
    def _update_sc_connectivity(conn, atid):
        """Update sidechain connectivity indices for a residue.
        Same as for the backbone but much simpler
        """
        result = []
        for idx in conn:
            result.append(atid + idx)
        return result
        
    def _check_connectivity(self, conn):
        """
        Check if the current bond is within the boundaries
        """
        for idx in conn:
            if idx < 1 or idx > self.natoms:
                return False
        return True

    def process_atoms(self, secstruc=[], start_atom=0, start_resid=1):
        """
        Makes a list of tuples representic atoms to convert after to GROMACS itp file
        FF input is given as (atid, type, name, chargegrp, charge, mass)
        We need to convert it to (atid, type, resid, resname, name, chargegrp, charge, mass, comment)
        """
        atid = start_atom
        resid = start_resid
        for resname in self.sequence:
            res_list = self.ff.bb_atoms + self.ff.sc_atoms(resname) # List of FF atoms in the residue
            reslen = len(res_list) # N atoms in the current residue
            for ffatom in res_list:
                atom_id = ffatom[0] + atid
                atom_type = ffatom[1]
                name = ffatom[2]
                chargegrp = ffatom[3] + atid
                charge = ffatom[4]
                mass = ffatom[5]
                comment = ""
                atom = [atom_id, atom_type, resid, resname, name, chargegrp, charge, mass, comment] # Need them mutable
                self.atoms.append(atom)
            prevreslen = reslen
            atid += reslen
            resid += 1
        self.atoms.pop(0)  # Remove the first atom
        self.natoms = len(self.atoms)

    def process_bb_bonds(self, secstruc=[], start_atom=0, start_resid=1):
        """
        This method handles the mapping and the creation of backbone connectivity.
        As is FF, bond must be a list of tuples [(connectivity), (parameters), (comment)]

        :param sequence: The nucleic acid sequence or a Chain instance.
        :param secstruc: Secondary structure information.
        """
        logging.debug(self.sequence)
        # Process backbone connectivity 
        atid = start_atom
        resid = start_resid
        prevreslen = None
        for resname in self.sequence:
            reslen = len(self.ff.bb_atoms) + len(self.ff.sc_atoms(resname)) # N atoms in the current residue
            ff_blist = self.ff.bb_blist # List with FF bonded types
            for btype, ffbtype in zip(self.blist, ff_blist): # Iterating through all types
                for bond in ffbtype: # Iterating through bonds in the type.
                # Updating the connectivity since FF connectivity given with respect to 0-th atom
                    if bond: # Don't interate through empty parameters.
                        connectivity = bond[0] 
                        parameters = bond[1]
                        comment = bond[2]              
                        upd_conn = self._update_bb_connectivity(connectivity, atid, reslen, prevreslen=prevreslen)
                        if self._check_connectivity(upd_conn): # Check if the current bond is within the boundaries
                            upd_bond = [list(upd_conn), list(parameters), comment] # Need them mutable
                            btype.append(upd_bond)
            prevreslen = reslen
            atid += reslen
            resid += 1

    def process_sc_bonds(self, secstruc=[], start_atom=0, start_resid=1):
        """
        This method handles the mapping and the creation of sidechain connectivity.
        As is FF, bond must be a list of tuples [(connectivity), (parameters), (comment)]

        :param sequence: The nucleic acid sequence or a Chain instance.
        :param secstruc: Secondary structure information.
        """
        # Process sidechain connectivity 
        atid = start_atom
        resid = start_resid
        prevreslen = None
        for resname in self.sequence:
            reslen = len(self.ff.bb_atoms) + len(self.ff.sc_atoms(resname)) # N atoms in the current residue
            ff_blist = self.ff.sc_blist(resname) # List with FF bonded types
            for btype, ffbtype in zip(self.blist, ff_blist): # Iterating through all types
                for bond in ffbtype: # Iterating through bonds in the type.
                # Updating the connectivity since FF connectivity given with respect to 0-th atom
                    if bond: # Don't interate through empty parameters.
                        connectivity = bond[0] 
                        parameters = bond[1]  
                        comment = bond[2]               
                        upd_conn = self._update_sc_connectivity(connectivity, atid)
                        if self._check_connectivity(upd_conn): # Check if the current bond is within the boundaries
                            upd_bond = [list(upd_conn), list(parameters), comment] # Need them mutable
                            btype.append(upd_bond)
            prevreslen = reslen
            atid += reslen
            resid += 1
        logging.info("Finished nucleic acid topology construction.")


    def elastic_network(self, atoms, anames=['BB1', 'BB3',], el=0.5, eu=1.2, ef=200):
        atoms = [atom for atom in atoms if atom.name in anames]
        for a1 in atoms:
            for a2 in atoms:
                if a2.atid - a1.atid > 3:
                    v1 = np.array((a1.x, a1.y, a1.z))
                    v2 = np.array((a2.x, a2.y, a2.z))
                    d = get_distance(v1, v2)
                    if d > el and d < eu:
                        comment = a1.resname + str(a1.resid) + "-" + a2.resname + str(a2.resid)
                        self.elnet.append([[a1.atid, a2.atid], [6, d, ef], comment])
        

