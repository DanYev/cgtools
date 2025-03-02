#!/usr/bin/env python3
"""Topology Module

Description:
    This module provides classes and functions to construct a coarse-grained
    topology from force field parameters. It defines the Topology class along with
    a helper BondList class. Methods are provided to process atoms, bonds, and
    connectivity information, and to generate topology files for coarse-grained
    simulations.

Usage Example:
    >>> from topology import Topology
    >>> from reforge.forcefield import NucleicForceField
    >>> ff = NucleicForceField(...)  # Initialize the force field instance
    >>> topo = Topology(ff, sequence=['A', 'T', 'G', 'C'])
    >>> topo.from_sequence(['A', 'T', 'G', 'C'])
    >>> topo.write_to_itp('output.itp')

Requirements:
    - Python 3.x
    - NumPy
    - reForge utilities and force field modules

Author: DY
Date: YYYY-MM-DD
"""

import logging
from typing import List
import numpy as np
from reforge import itpio

############################################################
# Helper class for working with bonds
############################################################

class BondList(list):
    """BondList Class

    Description:
        A helper subclass of the built-in list for storing bond information.
        Each element represents a bond in the form [connectivity, parameters, comment].

    Attributes
    ----------
    (inherited from list)
        Holds individual bond representations.

    Usage Example:
        >>> bonds = BondList([['C1-C2', [1.0, 1.5], 'res1 bead1'],
        ...                   ['C2-O1', [1.1, 1.6], 'res2 bead2']])
        >>> print(bonds.conns)
        ['C1-C2', 'C2-O1']
        >>> bonds.conns = ['C1-C2_mod', 'C2-O1_mod']
        >>> print(bonds.conns)
        ['C1-C2_mod', 'C2-O1_mod']
    """

    def __add__(self, other):
        """Implement addition of two BondList objects.

        Returns
        -------
        BondList
            A new BondList containing the bonds from self and other.
        """
        return BondList(super().__add__(other))

    @property
    def conns(self):
        """Get connectivity values from each bond.

        Returns
        -------
        list
            List of connectivity values (index 0 of each bond).
        """
        return [bond[0] for bond in self]

    @conns.setter
    def conns(self, new_conns):
        """Set new connectivity values for each bond.

        Parameters
        ----------
        new_conns : iterable
            A list-like object of new connectivity values. Must match the length of the BondList.
        """
        if len(new_conns) != len(self):
            raise ValueError("Length of new connectivity list must match the number of bonds")
        for i, new_conn in enumerate(new_conns):
            bond = list(self[i])
            bond[0] = new_conn
            self[i] = bond

    @property
    def params(self):
        """Get parameters from each bond.

        Returns
        -------
        list
            List of parameter values (index 1 of each bond).
        """
        return [bond[1] for bond in self]

    @params.setter
    def params(self, new_params):
        """Set new parameter values for each bond.

        Parameters
        ----------
        new_params : iterable
            A list-like object of new parameter values. Must match the length of the BondList.
        """
        if len(new_params) != len(self):
            raise ValueError("Length of new parameters list must match the number of bonds")
        for i, new_param in enumerate(new_params):
            bond = list(self[i])
            bond[1] = new_param
            self[i] = bond

    @property
    def comms(self):
        """Get comments from each bond.

        Returns
        -------
        list
            List of comments (index 2 of each bond).
        """
        return [bond[2] for bond in self]

    @comms.setter
    def comms(self, new_comms):
        """Set new comments for each bond.

        Parameters
        ----------
        new_comms : iterable
            A list-like object of new comment values. Must match the length of the BondList.
        """
        if len(new_comms) != len(self):
            raise ValueError("Length of new comments list must match the number of bonds")
        for i, new_comm in enumerate(new_comms):
            bond = list(self[i])
            bond[2] = new_comm
            self[i] = bond

    @property
    def measures(self):
        """Get measure values from each bond.

        Returns
        -------
        list
            List of measures extracted from the second element of the bond parameters.
        """
        return [bond[1][1] for bond in self]

    @measures.setter
    def measures(self, new_measures):
        """Set new measure values for each bond.

        Parameters
        ----------
        new_measures : iterable
            A list-like object of new measure values. Must match the length of the BondList.
        """
        if len(new_measures) != len(self):
            raise ValueError("Length of new measures list must match the number of bonds")
        for i, new_measure in enumerate(new_measures):
            bond = list(self[i])
            param = list(bond[1])
            param[1] = new_measure
            bond[1] = param
            self[i] = bond

    def categorize(self):
        """Categorize bonds based on their comments.

        Description:
            Uses the stripped comment (index 2) as a key to group bonds into a dictionary.

        Returns
        -------
        dict
            A dictionary mapping each unique comment to a BondList of bonds.
        """
        keys = sorted({comm.strip() for comm in self.comms})
        adict = {key: BondList() for key in keys}
        for bond in self:
            key = bond[2].strip()
            adict[key].append(bond)
        return adict

    def filter(self, condition, bycomm=True):
        """Filter bonds based on a provided condition.

        Description:
            Selects bonds for which the condition (a callable) returns True.
            By default, the condition is applied to the comment field.

        Parameters
        ----------
        condition : callable
            A function that takes a bond (or its comment) as input and returns True if the bond should be included.
        bycomm : bool, optional
            If True, the condition is applied to the comment (default is True).

        Returns
        -------
        BondList
            A new BondList containing the bonds that meet the condition.
        """
        if bycomm:
            return BondList([bond for bond in self if condition(bond[2])])
        return BondList([bond for bond in self if condition(bond)])


############################################################
# Topology Class
############################################################

class Topology:
    """Topology Class

    Description:
        Constructs a coarse-grained topology from force field parameters.
        Provides methods for processing atoms, bonds, and connectivity, and for
        generating a topology file for coarse-grained simulations.

    Attributes
    ----------
    ff : object
        Force field instance.
    sequence : list
        List of residue names.
    name : str
        Molecule name.
    nrexcl : int
        Exclusion parameter.
    atoms : list
        List of atom records.
    bonds, angles, dihs, cons, excls, pairs, vs3s, posres, elnet : BondList
        BondList instances for various bonded interactions.
    blist : list
        List containing all bond-type BondLists.
    secstruct : list
        Secondary structure as a list of characters.
    natoms : int
        Total number of atoms.
    """

    def __init__(self, forcefield, sequence: List = None, secstruct: List = None, **kwargs) -> None:
        """Initialize a Topology instance.

        Description:
            Initialize a topology with force field parameters, sequence, and optional secondary
            structure. Atom records and bond lists are initialized here.

        Parameters
        ----------
        forcefield : object
            An instance of the nucleic force field class.
        sequence : list, optional
            List of residue names.
        secstruct : list, optional
            Secondary structure as a list of characters. If not provided, defaults to 'F' for each residue.
        **kwargs :
            Additional keyword arguments. Recognized options include:
                molname : str
                    Molecule name (default: "molecule").
                nrexcl : int
                    Exclusion parameter (default: 1).
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
        self.blist = [self.bonds, self.angles, self.dihs, self.cons, self.excls, self.pairs, self.vs3s]
        self.secstruct = secstruct if secstruct is not None else ["F"] * len(self.sequence)

    def __iadd__(self, other) -> "Topology":
        """Implement in-place addition of another Topology instance.

        Description:
            Merges another Topology into this one by updating atom numbers and connectivity.

        Parameters
        ----------
        other : Topology
            Another Topology instance to merge with.

        Returns
        -------
        Topology
            The merged topology (self).
        """
        def update_atom(atom, atom_shift, residue_shift):
            atom[0] += atom_shift  # Update atom id
            atom[2] += residue_shift  # Update residue id
            atom[5] += atom_shift  # Update charge group number
            return atom

        def update_bond(bond, atom_shift):
            conn = bond[0]
            conn = [idx + atom_shift for idx in conn]
            return [conn, bond[1], bond[2]]

        atom_shift = self.natoms
        residue_shift = len(self.sequence)
        new_atoms = [update_atom(atom, atom_shift, residue_shift) for atom in other.atoms]
        self.atoms.extend(new_atoms)
        for self_attrib, other_attrib in zip(self.blist, other.blist):
            updated_bonds = [update_bond(bond, atom_shift) for bond in other_attrib]
            self_attrib.extend(updated_bonds)
        return self

    def __add__(self, other) -> "Topology":
        """Implement addition of two Topology objects.

        Description:
            Returns a new Topology that is the merger of self and other.

        Parameters
        ----------
        other : Topology
            Another Topology instance.

        Returns
        -------
        Topology
            A new Topology instance resulting from the merge.
        """
        new_top = self
        new_top += other
        return new_top

    def lines(self) -> list:
        """Generate the topology file as a list of lines.

        Returns
        -------
        list
            A list of strings, each representing a line in the topology file.
        """
        lines = itpio.format_header(molname=self.name, forcefield=self.ff.name, arguments="")
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

    def write_to_itp(self, filename: str):
        """Write the topology to an ITP file.

        Parameters
        ----------
        filename : str
            The output file path.
        """
        with open(filename, "w", encoding="utf-8") as file:
            for line in self.lines():
                file.write(line)

    @staticmethod
    def _update_bb_connectivity(conn, atid, reslen, prevreslen=None):
        """Update backbone connectivity indices for a residue.

        Description:
            Adjusts atom indices based on the length of the current residue and, if provided,
            the previous residue for dihedral definitions.

        Parameters
        ----------
        conn : list of int
            Connectivity indices from the force field. Negative indices indicate connections
            relative to the previous residue.
        atid : int
            Atom ID of the first atom in the current residue.
        reslen : int
            Number of atoms in the current residue.
        prevreslen : int, optional
            Number of atoms in the previous residue. If None, negative indices are not updated.

        Returns
        -------
        list
            A list of updated connectivity indices.

        Example
        -------
        >>> conn = [0, 1, -1]
        >>> Topology._update_bb_connectivity(conn, 10, 5, prevreslen=4)
        [10, 11, 8]
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

        Description:
            Adjusts sidechain connectivity by adding the starting atom index.

        Parameters
        ----------
        conn : list of int
            Connectivity indices for the sidechain.
        atid : int
            Starting atom id for the residue.

        Returns
        -------
        list
            A list of updated connectivity indices.
        """
        return [atid + idx for idx in conn]

    def _check_connectivity(self, conn):
        """Check if the connectivity indices are within valid boundaries.

        Parameters
        ----------
        conn : list of int
            Connectivity indices to check.

        Returns
        -------
        bool
            True if all indices are between 1 and natoms, False otherwise.
        """
        for idx in conn:
            if idx < 1 or idx > self.natoms:
                return False
        return True

    def process_atoms(self, start_atom: int = 0, start_resid: int = 1):
        """Process atoms based on the sequence and force field definitions.

        Description:
            For each residue in the sequence, constructs atom records using both
            backbone and sidechain definitions from the force field.

        Parameters
        ----------
        start_atom : int, optional
            Starting atom ID (default is 0).
        start_resid : int, optional
            Starting residue ID (default is 1).
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
        """Process backbone bonds using force field definitions.

        Parameters
        ----------
        start_atom : int, optional
            Starting atom ID.
        start_resid : int, optional
            Starting residue ID.
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
        """Process sidechain bonds using force field definitions.

        Parameters
        ----------
        start_atom : int, optional
            Starting atom ID.
        start_resid : int, optional
            Starting residue ID.
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
        """Construct an elastic network between selected atoms.

        Parameters
        ----------
        atoms : list
            List of atom objects.
        anames : list of str, optional
            Atom names to include (default: ["BB1", "BB3"]).
        el : float, optional
            Lower distance cutoff.
        eu : float, optional
            Upper distance cutoff.
        ef : float, optional
            Force constant.
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
        """Build topology from a given sequence.

        Parameters
        ----------
        sequence : list
            Nucleic acid sequence (list of residue names).
        secstruc : list, optional
            Secondary structure. If not provided, defaults to all 'F'.
        """
        self.sequence = sequence
        self.process_atoms()
        self.process_bb_bonds()
        self.process_sc_bonds()

    def from_chain(self, chain, secstruc=None):
        """Build topology from a chain instance.

        Parameters
        ----------
        chain : object
            Chain object containing residues.
        secstruc : list, optional
            Secondary structure.
        """
        sequence = [residue.resname for residue in chain]
        self.from_sequence(sequence, secstruc=secstruc)

    @staticmethod
    def merge_topologies(topologies):
        """Merge multiple Topology instances into one.

        Parameters
        ----------
        topologies : list
            List of Topology objects.

        Returns
        -------
        Topology
            A merged Topology instance.
        """
        top = topologies.pop(0)
        if topologies:
            for new_top in topologies:
                top += new_top
        return top
