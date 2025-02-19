#!/usr/bin/env python3
"""
Refactored Topology module.

This module contains the Topology class with improved readability,
modularity, and error checking. Many long methods have been split into
helper functions to improve maintainability.
"""

import logging
from cgtools.forge.forcefields import NucleicForceField
from typing import Any, Dict, List, Optional, Tuple, Union

# NOTE: The following classes (CategorizedList, Chain, Pair, Bond, Angle, Dihedral,
# Vsite, Exclusion, CoarseGrained, etc.) and constants (ElasticMaximumForce,
# ElasticLowerBound, ElasticUpperBound, ElasticDecayFactor, ElasticDecayPower,
# ElasticMinimumForce, ElasticBeads, enStrandLengths) are assumed to be defined
# elsewhere in your code base.

class Topology:
    def __init__(self, forcefield, sequence: List = [], secstruct: List = []) -> None:
        """
        Initialize a Topology instance.

        forcefield - an instance of NucleicForceField Class
        :param other: Another Topology or Chain instance to initialize from.
        :param options: Dictionary of options, e.g. ForceField, Version, etc.
        :param name: Optional name for the topology.
        """
        self.ff = forcefield
        self.sequence = sequence
        self.name: str = ''
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
        self.mapping: List = []
        # list with all bonded parameters like in the ff
        self.blist = [self.bonds, self.angles, self.dihs, self.cons, 
            self.excls, self.pairs, self.vs3s]  
        # Secondary structure     
        if secstruct:
            self.secstruct = secstruct
        else:
            self.secstruct = []


    def __iadd__(self, other: Any) -> "Topology":
        """
        Implements in-place addition of another Topology.

        :param other: Another Topology instance or object convertible to one.
        :return: self after merging the two topologies.
        """
        if not isinstance(other, Topology):
            other = Topology(other)
        shift = len(self.atoms)
        last_atom = self.atoms[-1]
        # Update atoms
        new_atoms = list(zip(*other.atoms))
        new_atoms[0] = [atom_num + shift for atom_num in new_atoms[0]]   # Update atom numbers
        new_atoms[2] = [resi + last_atom[2] for resi in new_atoms[2]]       # Update residue numbers
        new_atoms[5] = [cg + last_atom[5] for cg in new_atoms[5]]             # Update charge group numbers
        new_atoms = list(zip(*new_atoms))
        new_atoms = [self._fix_atom_record(atom) for atom in new_atoms]
        self.atoms.extend(new_atoms)

        # Update other categories
        for attrib in ["bonds", "vsites", "exclusions", "angles", "dihedrals", "impropers", "constraints", "posres"]:
            category = getattr(self, attrib)
            other_category = getattr(other, attrib)
            # Shift indices in each item
            for item in other_category:
                item += shift
            category.extend(other_category)
        return self

    def __add__(self, other: Any) -> "Topology":
        """
        Implements addition of two Topology objects.

        :param other: Another Topology or object convertible to Topology.
        :return: A new Topology instance that is the sum of self and other.
        """
        new_topology = Topology(self)
        new_topology += other
        return new_topology


    def __str__(self) -> str:
        """
        Returns a string representation of the topology file.
        """
        sections = [
            self._format_header(),
            self._format_sequence_section(),
            self._format_moleculetype_section(),
            self._format_atoms_section(),
            self._format_pairs_section(),
            self._format_virtual_sites_section(),
            self._format_bonds_section(),
            self._format_constraints_section(),
            self._format_exclusions_section(),
            self._format_angles_section(),
            self._format_dihedrals_section(),
            self._format_posres_section()
        ]
        logging.info('Created coarsegrained topology')
        return "\n".join([sec for sec in sections if sec])


    def _format_sequence_section(self) -> str:
        """
        Formats the sequence section if present.
        """
        if not self.sequence:
            return ""
        seq_str = '; Sequence:\n; ' + ''.join([AA321.get(AA) for AA in self.sequence])
        secstruc_str = '; Secondary Structure:\n; ' + self.secstruc
        return "\n".join([seq_str, secstruc_str])


    def _format_bonds_section(self) -> str:
        """
        Formats the bonds section.
        """
        lines = ["\n[ bonds ]"]
        # Backbone-backbone bonds
        bb_bonds = [str(bond) for bond in self.bonds["BB"] if str(bond)]
        if bb_bonds:
            lines.append("; Backbone bonds")
            lines.extend(bb_bonds)
        # Rubber bands with CPP directives if present
        rubber_bonds = [str(bond) for bond in self.bonds.get(("Rubber", True), []) if str(bond)]
        if rubber_bonds:
            try:
                elastic_force = self.options['ElasticMaximumForce']
            except KeyError:
                elastic_force = 0.0
            lines.append("#ifdef RUBBER_BANDS")
            lines.append("#ifndef RUBBER_FC\n#define RUBBER_FC %f\n#endif" % elastic_force)
            lines.extend(rubber_bonds)
            lines.append("#endif")
        # Sidechain bonds
        sc_bonds = [str(bond) for bond in self.bonds["SC"] if str(bond)]
        if sc_bonds:
            lines.append("; Sidechain bonds")
            lines.extend(sc_bonds)
        # Elastic bonds
        for label, comment in [("Elastic short", "; Short elastic bonds for extended regions"),
                               ("Elastic long", "; Long elastic bonds for extended regions")]:
            bonds = [str(bond) for bond in self.bonds[label] if str(bond)]
            if bonds:
                lines.append(comment)
                lines.extend(bonds)
        # Links
        for label, comment in [("Link", "; Links")]:
            bonds = [str(bond) for bond in self.bonds[label] if str(bond)]
            if bonds:
                lines.append(comment)
                lines.extend(bonds)
        return "\n".join(lines)


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
            tuple: A tuple of updated connectivity indices.

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
        return tuple(result)

    @staticmethod
    def _update_sc_connectivity(conn, atid):
        """Update sidechain connectivity indices for a residue.
        Same as for the backbone but much simpler
        """
        result = []
        for idx in conn:
            result.append(atid + idx)
        return tuple(result)
        
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
                comment = ()
                atom = (atom_id, atom_type, resid, resname, name, chargegrp, charge, mass, comment)
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
                            upd_bond = [upd_conn, parameters, comment]
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
                            upd_bond = [upd_conn, parameters, comment]
                            btype.append(upd_bond)
            prevreslen = reslen
            atid += reslen
            resid += 1
        logging.info("Finished nucleic acid topology construction.")
        

