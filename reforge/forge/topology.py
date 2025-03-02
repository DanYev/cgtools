#!/usr/bin/env python3
"""Topology module.

This module contains the Topology class with improved readability,
modularity, and error checking. Many long methods have been split into
helper functions to improve maintainability.
"""

import logging
from typing import List
import numpy as np
from reforge import itpio

############################################################
# Helper class to work with "bonds" - list of bonds.
# Defined later in Topology calss
# A bond is [connectivity, parameters, comment]
############################################################

class BondList(list):
    """BondList is a helper subclass of the built-in list, specialized for
    storing bond information. Each element in a BondList is expected to
    represent a bond (e.g., a list or tuple of atoms connected by the bond),
    namely [connectivity, parameters, comment].

    Attributes:
        (inherited from list) The list holds individual bond representations.

    Example:
        >>> bonds = BondList([['C1-C2', [1.0, 1.5], 'res1 bead1'], ['C2-O1', [1.1, 1.6], 'res2 bead2']])
        >>> print(bonds.conns)
        ['C1-C2', 'C2-O1']
        >>> bonds.conns = ['C1-C2_mod', 'C2-O1_mod']
        >>> print(bonds.conns)
        ['C1-C2_mod', 'C2-O1_mod']
    """

    def __add__(self, other):
        """Implements addition of two BondList objects."""
        return BondList(super().__add__(other))  # BondList(list(self) + list(other))

    @property
    def conns(self):
        """Returns a list of connectivity values extracted from each bond
        (assumed to be at index 0)."""
        return [bond[0] for bond in self]

    @conns.setter
    def conns(self, new_conns):
        """Sets the connectivity for each bond.

        Parameters:
            new_conns (iterable): A list-like object of new connectivity values.
                                  Must have the same length as the BondList.
        """
        if len(new_conns) != len(self):
            raise ValueError(
                "Length of new connectivity list must match the number of bonds"
            )
        for i, new_conn in enumerate(new_conns):
            # Convert the bond to a mutable list if needed
            bond = list(self[i])
            bond[0] = new_conn
            self[i] = bond

    @property
    def params(self):
        """Returns a list of parameters extracted from each bond (assumed to be
        at index 1)."""
        return [bond[1] for bond in self]

    @params.setter
    def params(self, new_params):
        """Sets the parameters for each bond.

        Parameters:
            new_params (iterable): A list-like object of new parameter values.
                                   Must have the same length as the BondList.
        """
        if len(new_params) != len(self):
            raise ValueError(
                "Length of new parameters list must match the number of bonds"
            )
        for i, new_param in enumerate(new_params):
            bond = list(self[i])
            bond[1] = new_param
            self[i] = bond

    @property
    def comms(self):
        """Returns a list of comments extracted from each bond (assumed to be
        at index 2)."""
        return [bond[2] for bond in self]

    @comms.setter
    def comms(self, new_comms):
        """Sets the comments for each bond.

        Parameters:
            new_comms (iterable): A list-like object of new comment values.
                                  Must have the same length as the BondList.
        """
        if len(new_comms) != len(self):
            raise ValueError(
                "Length of new comments list must match the number of bonds"
            )
        for i, new_comm in enumerate(new_comms):
            bond = list(self[i])
            bond[2] = new_comm
            self[i] = bond

    @property
    def measures(self):
        """Returns a list of 'measures' extracted from each bond.

        Assumes that each bond's parameter is an iterable where the
        measure is at index 1.
        """
        return [bond[1][1] for bond in self]

    @measures.setter
    def measures(self, new_measures):
        """Sets the measure (assumed to be the second element in the
        parameters) for each bond.

        Parameters:
            new_measures (iterable): A list-like object of new measure values.
                                     Must have the same length as the BondList.
        """
        if len(new_measures) != len(self):
            raise ValueError(
                "Length of new measures list must match the number of bonds"
            )
        for i, new_measure in enumerate(new_measures):
            bond = list(self[i])
            param = list(bond[1])
            param[1] = new_measure
            bond[1] = param
            self[i] = bond

    def categorize(self):
        """Categorize bonds based on their comments.

        Typically, the first string in the comment is used as a key.
        Returns:
            dict: A dictionary mapping each unique comment (stripped of whitespace) 
            to a BondList of bonds with that comment.
        """
        keys = [comm.strip() for comm in set(self.comms)]
        keys = sorted(keys)
        adict = {key: BondList() for key in keys}
        for bond in self:
            key = bond[2].strip()  # bond[2] is the comment
            adict[key].append(bond)
        return adict

    def filter(self, condition, bycomm=True):
        """Bycomm (bool): if filter by comment Select bonds based on a generic
        condition condition (callable): A function that takes a bond as input
        and returns True if the bond should be included, False otherwise."""
        return BondList([bond for bond in self if condition(bond[2])])


class Topology:
    """
    Constructs a coarse-grained topology from force field parameters.

    Attributes:
        ff: Force field instance.
        sequence: List of residue names.
        name: Molecule name.
        nrexcl: Exclusion parameter.
        atoms: List of atom records.
        bonds, angles, dihs, cons, excls, pairs, vs3s, posres, elnet: BondList instances.
        blist: List containing all bond-type BondLists.
        secstruct: Secondary structure as a list of characters.
        natoms: Total number of atoms.
    """
    def __init__(
        self, forcefield, sequence: List=None, secstruct: List=None, **kwargs
    ) -> None:
        """Initialize a Topology instance. Main attributes are:

        1. atom - [atid, type, resid, resname, name, chargegrp, charge, mass, comment]
        2. bond - [connectivity, parameters, comment]

        forcefield - an instance of NucleicForceField Class
        :param other: Another Topology or Chain instance to initialize from.
        :param options: Dictionary of options, e.g. ForceField, Version, etc.
        :param name: Optional name for the topology.
        """
        molname = kwargs.pop("molname", "molecule")
        nrexcl = kwargs.pop("nrexcl", 1)
        self.ff = forcefield
        self.sequence = sequence
        self.name = molname
        self.nrexcl = nrexcl
        self.atoms: List = []
        self.bonds = BondList()
        self.angles = BondList()
        self.dihs = BondList()
        self.cons = BondList()
        self.excls = BondList()
        self.pairs = BondList()
        self.vs3s = BondList()
        self.posres = BondList()
        self.elnet = BondList()
        self.mapping: List = []
        self.natoms = len(self.atoms)
        # list with all bonded parameters as in the ff
        self.blist = [
            self.bonds,
            self.angles,
            self.dihs,
            self.cons,
            self.excls,
            self.pairs,
            self.vs3s,
        ]
        # Secondary structure
        self.secstruct = secstruct if secstruct is not None else ["F"] * len(self.sequence)

    def __iadd__(self, other) -> "Topology":
        """Implements in-place addition of another Topology.

        1. atom - (atid, type, resid, resname, name, chargegrp, charge, mass, comment)
        2. bond - [connectivity, parameters, comment]
        :param other: Another Topology instance or object convertible to one.
        :return: self after merging the two topologies.
        """

        def update_atom(atom, atom_shift, residue_shift):
            atom[0] += atom_shift  # Update atom numbers
            atom[2] += residue_shift  # Update residue numbers
            atom[5] += atom_shift  # Update charge group numbers
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
        """Implements addition of two Topology objects."""
        # Create a copy of self. Assumes that the constructor can create a copy from self.
        new_top = self
        new_top += other  # Use __iadd__ to add the other topology
        return new_top

    def lines(self) -> list:
        """Returns the topology file representation as a list of lines.

        Returns:
            list: A list of strings, where each string is a line in the topology file.
        """
        lines = itpio.format_header(
            molname=self.name, forcefield=self.ff.name, arguments=""
        )
        lines += itpio.format_sequence_section(self.sequence, self.secstruct)
        lines += itpio.format_moleculetype_section(molname=self.name, nrexcl=1)
        lines += itpio.format_atoms_section(self.atoms)
        lines += itpio.format_bonded_section("bonds", self.bonds)
        lines += itpio.format_bonded_section("angles", self.angles)
        lines += itpio.format_bonded_section("dihedrals", self.dihs)
        lines += itpio.format_bonded_section("constraints", self.cons)
        lines += itpio.format_bonded_section("exclusions", self.excls)
        lines += itpio.format_bonded_section("pairs", self.pairs)
        lines += itpio.format_bonded_section("virtual_sites3", self.vs3s)
        lines += itpio.format_bonded_section("bonds", self.elnet)
        lines += itpio.format_posres_section(self.atoms)
        logging.info("Created coarsegrained topology")
        return lines

    # def __str__(self) -> str:
    #     return "".join(self.lines())

    def write_to_itp(self, filename: str):
        """Write the topology lines to a file."""
        with open(filename, "w", encoding="utf-8") as file:
            for line in self.lines():
                file.write(line)

    @staticmethod
    def _update_bb_connectivity(conn, atid, reslen, prevreslen=None):
        """Update backbone connectivity indices for a residue.

        This method updates backbone connectivity indices provided by the force field (FF) 
        for the current residue. It adjusts atom indices based on the length of the current 
        residue, and for some dihedral definitions, uses the length of the previous residue.

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
                if prevreslen is not None:
                    result.append(atid - prevreslen + idx + 3)
                    continue
                return list(conn)
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
        """Check if the current bond is within the boundaries."""
        for idx in conn:
            if idx < 1 or idx > self.natoms:
                return False
        return True

    def process_atoms(self, start_atom: int = 0, start_resid: int = 1):
        """
        Process atoms from the sequence based on force field definitions.

        Args:
            start_atom (int): Starting atom id (default 0).
            start_resid (int): Starting residue id (default 1).
        """
        atid = start_atom
        resid = start_resid
        for resname in self.sequence:
            ff_atoms = self.ff.bb_atoms + self.ff.sc_atoms(resname)
            reslen = len(ff_atoms)
            for ffatom in ff_atoms:
                atom = [
                    ffatom[0] + atid,    # atom id
                    ffatom[1],           # type
                    resid,               # residue id
                    resname,             # residue name
                    ffatom[2],           # name
                    ffatom[3] + atid,    # charge group
                    ffatom[4],           # charge
                    ffatom[5],           # mass
                    "",
                ]
                self.atoms.append(atom)
            atid += reslen
            resid += 1
        self.atoms.pop(0)  # Remove dummy atom
        self.natoms = len(self.atoms)

    def process_bb_bonds(self, start_atom: int = 0, start_resid: int = 1):
        """
        Process backbone bonds using force field definitions.

        Args:
            start_atom (int): Starting atom id.
            start_resid (int): Starting residue id.
        """
        logging.debug(self.sequence)
        atid = start_atom
        resid = start_resid
        prevreslen = None
        for resname in self.sequence:
            reslen = len(self.ff.bb_atoms) + len(self.ff.sc_atoms(resname))
            ff_blist = self.ff.bb_blist
            for btype, ff_btype in zip(self.blist, ff_blist):
                for bond in ff_btype:
                    if bond:
                        connectivity = bond[0]
                        parameters = bond[1]
                        comment = bond[2]
                        upd_conn = self._update_bb_connectivity(connectivity, atid, reslen, prevreslen)
                        if self._check_connectivity(upd_conn):
                            upd_bond = [list(upd_conn), list(parameters), comment]
                            btype.append(upd_bond)
            prevreslen = reslen
            atid += reslen
            resid += 1

    def process_sc_bonds(self, start_atom: int = 0, start_resid: int = 1):
        """
        Process sidechain bonds using force field definitions.

        Args:
            start_atom (int): Starting atom id.
            start_resid (int): Starting residue id.
        """
        atid = start_atom
        resid = start_resid
        for resname in self.sequence:
            reslen = len(self.ff.bb_atoms) + len(self.ff.sc_atoms(resname))
            ff_blist = self.ff.sc_blist(resname)
            for btype, ff_btype in zip(self.blist, ff_blist):
                for bond in ff_btype:
                    if bond:
                        connectivity = bond[0]
                        parameters = bond[1]
                        comment = bond[2]
                        upd_conn = self._update_sc_connectivity(connectivity, atid)
                        if self._check_connectivity(upd_conn):
                            upd_bond = [list(upd_conn), list(parameters), comment]
                            btype.append(upd_bond)
            atid += reslen
            resid += 1
        logging.info("Finished nucleic acid topology construction.")

    def elastic_network(self, atoms, anames: List[str] = None, el: float = 0.5, eu: float = 1.2, ef: float = 200):
        """
        Construct an elastic network between selected atoms.

        Args:
            atoms: List of atom objects.
            anames (List[str]): Atom names to include (default: ["BB1", "BB3"]).
            el (float): Lower distance cutoff.
            eu (float): Upper distance cutoff.
            ef (float): Force constant.
        """
        if anames is None:
            anames = ["BB1", "BB3"]
        def get_distance(v1, v2):
            return np.linalg.norm(np.array(v1) - np.array(v2)) / 10.0

        selected = [atom for atom in atoms if atom[4] in anames]
        for a1 in selected:
            for a2 in selected:
                if a2[0] - a1[0] > 3:
                    v1 = (a1[5], a1[6], a1[7])
                    v2 = (a2[5], a2[6], a2[7])
                    d = get_distance(v1, v2)
                    if el < d < eu:
                        comment = f"{a1[3]}{a1[2]}-{a2[3]}{a2[2]}"
                        self.elnet.append([[a1[0], a2[0]], [6, d, ef], comment])

    def from_sequence(self, sequence, secstruc=None):
        """
        Build topology from a given sequence.

        Args:
            sequence: Nucleic acid sequence (list of residue names).
            secstruc (List): Secondary structure (default: all 'F').
        """
        self.sequence = sequence
        self.process_atoms()  # Adds itp atom objects to the topology list
        self.process_bb_bonds()  # Adds bb bond objects to the topology list
        self.process_sc_bonds()  # Adds sc bond objects to the topology list

    def from_chain(self, chain, secstruc=None):
        """
        Build topology from a chain instance.

        Args:
            chain: Chain object.
            secstruc (List): Secondary structure.
        """
        sequence = [
            residue.resname for residue in chain
        ]  # So far only need sequence for the topology
        self.from_sequence(sequence, secstruc=secstruc)

    @staticmethod
    def merge_topologies(topologies):
        """
        Merge multiple Topology instances into one.

        Args:
            topologies: List of Topology objects.

        Returns:
            Topology: Merged topology.
        """
        top = topologies.pop(0)
        if topologies:
            for new_top in topologies:
                top += new_top
        return top
