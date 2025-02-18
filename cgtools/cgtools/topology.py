#!/usr/bin/env python3
"""
Refactored Topology module.

This module contains the Topology class with improved readability,
modularity, and error checking. Many long methods have been split into
helper functions to improve maintainability.
"""

import logging
from cgtools.cgtools.forcefields import NucleicForceField
from typing import Any, Dict, List, Optional, Tuple, Union

# NOTE: The following classes (CategorizedList, Chain, Pair, Bond, Angle, Dihedral,
# Vsite, Exclusion, CoarseGrained, etc.) and constants (ElasticMaximumForce,
# ElasticLowerBound, ElasticUpperBound, ElasticDecayFactor, ElasticDecayPower,
# ElasticMinimumForce, ElasticBeads, enStrandLengths) are assumed to be defined
# elsewhere in your code base.

class Topology:
    def __init__(self, forcefield, sequence: List = []) -> None:
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
        self.pairs: List = []
        self.vsites: List = []
        self.exclusions: List = []
        self.angles: List = []
        self.dihedrals: List = []
        self.impropers: List = []
        self.constraints: List = []
        self.posres: List = []
        self.secstruc: str = ""
        self.mapping: List = []
        self.breaks: List = []
        self.natoms: int = 0        


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
        new_topo = Topology(self)
        new_topo += other
        return new_topo

    def _fix_atom_record(self, atom: Tuple) -> Tuple:
        """
        Fixes atom record if mass is not specified.

        :param atom: Atom tuple.
        :return: Fixed atom tuple.
        """
        if atom[-1] == 0:
            return atom + ('c',)
        return atom

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

    def _format_header(self) -> str:
        """
        Formats the header of the topology file.
        """
        try:
            forcefield_name = self.options['ForceField'].name
            version = self.options['Version']
            arguments = ' '.join(self.options['Arguments'])
        except KeyError as e:
            logging.error("Missing key in options: %s", e)
            forcefield_name = "Unknown"
            version = "Unknown"
            arguments = ""
        header = f"; MARTINI ({forcefield_name}) Coarse Grained topology file for \"{self.name}\"\n"
        header += f"; Created by py version {version} \n; Using the following options: {arguments}\n"
        header += "; " + "#" * 100 + "\n"
        header += "; This topology is based on development beta of Martini DNA and should NOT be used for production runs.\n"
        header += "; " + "#" * 100
        return header

    def _format_sequence_section(self) -> str:
        """
        Formats the sequence section if present.
        """
        if not self.sequence:
            return ""
        seq_str = '; Sequence:\n; ' + ''.join([AA321.get(AA) for AA in self.sequence])
        secstruc_str = '; Secondary Structure:\n; ' + self.secstruc
        return "\n".join([seq_str, secstruc_str])

    def _format_moleculetype_section(self) -> str:
        """
        Formats the moleculetype section.
        """
        return "\n[ moleculetype ]\n; Name         Exclusions\n{:<15s} {:3d}".format(self.name, self.nrexcl)

    def _format_atoms_section(self) -> str:
        """
        Formats the atoms section.
        """
        out = ["\n[ atoms ]"]
        fs8 = '%5d %5s %5d %5s %5s %5d %7.4f ; %s'
        fs9 = '%5d %5s %5d %5s %5s %5d %7.4f %7.4f ; %s'
        for atom in self.atoms:
            formatted = fs9 % atom if len(atom) == 9 else fs8 % atom
            out.append(formatted)
        return "\n".join(out)

    def _format_pairs_section(self) -> str:
        """
        Formats the pairs section.
        """
        pairs = [str(pair) for pair in self.pairs if str(pair)]
        if pairs:
            return "\n[ pairs ]\n" + "\n".join(pairs)
        return ""

    def _format_virtual_sites_section(self) -> str:
        """
        Formats the virtual sites section.
        """
        vs_bb = [str(v) for v in self.vsites["BB"] if str(v)]
        vs_sc = [str(v) for v in self.vsites["SC"] if str(v)]
        if not (vs_bb or vs_sc):
            return ""
        lines = ["\n[ virtual_sites3 ]"]
        if vs_bb:
            lines.append("; Backbone virtual sites.")
            lines.extend(vs_bb)
        if vs_sc:
            lines.append("; Sidechain virtual sites.")
            lines.extend(vs_sc)
        return "\n".join(lines)

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

    def _format_constraints_section(self) -> str:
        """
        Formats the constraints section.
        """
        lines = ["\n[ constraints ]"]
        constraints = [str(constraint) for constraint in self.bonds["Constraint"]]
        lines.extend(constraints)
        return "\n".join(lines)

    def _format_exclusions_section(self) -> str:
        """
        Formats the exclusions section.
        """
        exclusions = [str(excl) for excl in self.exclusions if str(excl)]
        if exclusions:
            return "\n[ exclusions ]\n" + "\n".join(exclusions)
        return ""

    def _format_angles_section(self) -> str:
        """
        Formats the angles section.
        """
        lines = ["\n[ angles ]", "; Backbone angles"]
        lines.extend([str(angle) for angle in self.angles["BBB"] if str(angle)])
        lines.append("; Backbone-sidechain angles")
        lines.extend([str(angle) for angle in self.angles["BBS"] if str(angle)])
        lines.append("; Sidechain angles")
        lines.extend([str(angle) for angle in self.angles["SC"] if str(angle)])
        return "\n".join(lines)

    def _format_dihedrals_section(self) -> str:
        """
        Formats the dihedrals section.
        """
        lines = ["\n[ dihedrals ]", "; Backbone dihedrals"]
        lines.extend([str(dihed) for dihed in self.dihedrals["BBBB"] if dihed.parameters])
        lines.append("; Sidechain dihedrals")
        lines.extend([str(dihed) for dihed in self.dihedrals["BSC"] if dihed.parameters])
        lines.append("; Sidechain improper dihedrals")
        lines.extend([str(dihed) for dihed in self.dihedrals["SC"] if dihed.parameters])
        return "\n".join(lines)

    def _format_posres_section(self) -> str:
        """
        Formats the position restraints section.
        """
        if not self.posres:
            return ""
        try:
            posres_force = self.options['PosResForce']
        except KeyError:
            posres_force = 0.0
        lines = ["\n#ifdef POSRES",
                 "#ifndef POSRES_FC\n#define POSRES_FC %.2f\n#endif" % posres_force,
                 " [ position_restraints ]"]
        lines.extend(['  %5d    1    POSRES_FC    POSRES_FC    POSRES_FC' % idx for idx in self.posres])
        lines.append("#endif")
        return "\n".join(lines)

    @staticmethod
    def _update_bb_connectivity(conn, atid, reslen, prevreslen=None):
        result = []
        prev = -1
        for idx in conn:
            if idx < 0:
                result.append(atid - prevreslen + idx + 3) 
                continue
            if idx > prev:
                result.append(atid + idx)
            else:
                result.append(atid + idx + reslen)
            prev = idx
        return tuple(result)

    def process_bb_bonds(self, secstruc=[], start_atom=1, start_resi=1):
        """
        Constructs the topology from a nucleic acid sequence.
        This method handles the mapping and the creation of backbone and sidechain connectivity.

        :param sequence: The nucleic acid sequence or a Chain instance.
        :param secstruc: Secondary structure information.
        """
        # Log secondary structure and sequence information.
        print(self.sequence)
        logging.debug(self.sequence)
        logging.debug(secstruc)
        # Process backbone connectivity 
        atid = start_atom
        resid = start_resi
        prevreslen = None
        for resname in self.sequence:
            reslen = len(self.ff.bb_atoms) + len(self.ff.sc_atoms(resname))
            bonds = self.ff.bb_bonds
            for bond in bonds:
                conn = bond[0]
                params = bond[1]
                upd_conn = self._update_bb_connectivity(conn, atid, reslen, prevreslen=prevreslen)
                upd_bond = [upd_conn, params]
                self.bonds.append(upd_bond)
            prevreslen = reslen
            atid += reslen
            resid += 1

    def process_sc_bonds(self, secstruc=[], start_atom=1, start_resi=1):
        atid = startAtom
        resid = [resi for resi in resid for _ in range(3)]
        resnames = [res for res in self.sequence for _ in range(3)]
        scinfos = [item for item in sc for _ in range(3)]
        secstrucs = [s for s in self.secstruc for _ in range(3)]
        count = 0
        for resid, resname, bb_type, scinfo, ss in zip(resids, resnames, bb_type, scinfos, secstrucs):
            if (count % 3) == 0:
                (sc_atoms, bond_params, angle_params, dihed_params,
                 imp_params, vsite_params, excl_params, pair_params) = sidechain_info
                bon_conn, ang_conn, dih_conn, imp_conn, vsite_conn, excl_conn, pair_conn = \
                    (self.options['ForceField'].base_connectivity[resname] + 7 * [[]])[:7]
                for at_ids, par in zip(bon_conn, bond_params):
                    if par[2] == 100000.00000:
                        self.bonds.append(Bond(options=self.options, atoms=at_ids, parameters=[par[1]],
                                                type=par[0], comments=resname, category="Constraint"))
                    else:
                        self.bonds.append(Bond(options=self.options, atoms=at_ids, parameters=par[1:],
                                                type=par[0], comments=resname, category="SC"))
                    self.bonds[-1] += atid
                for at_ids, par in zip(ang_conn, angle_params):
                    self.angles.append(Angle(options=self.options, atoms=at_ids, parameters=par[1:],
                                             type=par[0], comments=resname, category="SC"))
                    self.angles[-1] += atid
                for at_ids, par in zip(dih_conn, dihed_params):
                    self.dihedrals.append(Dihedral(options=self.options, atoms=at_ids, parameters=par[1:],
                                                   type=par[0], comments=resname, category="BSC"))
                    self.dihedrals[-1] += atid
                for at_ids, par in zip(imp_conn, imp_params):
                    self.dihedrals.append(Dihedral(options=self.options, atoms=at_ids, parameters=par[1:],
                                                   type=par[0], comments=resname, category="SC"))
                    self.dihedrals[-1] += atid
                for at_ids, par in zip(vsite_conn, vsite_params):
                    self.vsites.append(Vsite(options=self.options, atoms=at_ids, parameters=par,
                                             comments=resname, category="SC"))
                    self.vsites[-1] += atid
                for at_ids, _ in zip(excl_conn, excl_params):
                    excl = Exclusion(options=self.options, atoms=at_ids, parameters=' ', category="SC")
                    self.exclusions.append(excl)
                    self.exclusions[-1] += atid
                for at_ids, _ in zip(pair_conn, pair_params):
                    pair = Pair(options=self.options, atoms=at_ids, parameters=' ')
                    self.pairs.append(pair)
                    self.pairs[-1] += atid

                counter = 0
                # Process backbone beads and sidechain atoms.
                bbb_set = [bbMulti[count], bbMulti[count+1], bbMulti[count+2]]
                for atype, aname in zip(bbb_set + list(sc_atoms), CoarseGrained.residue_bead_names_dna):
                    charge = self.options['ForceField'].getCharge(atype, aname)
                    if atid in [vSite.atoms[0] for vSite in self.vsites]:
                        mass = 0
                    else:
                        if aname == 'BB1':
                            mass = 72
                        elif aname in ("BB2", "BB3"):
                            mass = 60
                        else:
                            mass = 36
                    self.atoms.append((atid, atype, resi, resname, aname, atid, charge, mass, ss))
                    # Position restraints (@POSRES)
                    if 'all' in self.options.get('PosRes', []):
                        if aname in ("BB1", "BB2", "BB3", "SC1") and atid - 1 > 1:
                            self.posres.append(atid - 1)
                    if 'bb' in self.options.get('PosRes', []):
                        if aname == "BB2" and atid - 1 > 1:
                            self.posres.append(atid - 1)
                    if mapping:
                        self.mapping.append((atid, [i + shift for i in mapping[counter]]))
                    atid += 1
                    counter += 1
            count += 1

        # Clean up connectivity that may extend beyond the valid atom range.
        self._cleanup_connectivity()
        logging.info("Finished nucleic acid topology construction.")

    def _cleanup_connectivity(self) -> None:
        """
        Remove connectivity entries (dihedrals, angles, bonds, etc.) that reference
        non-existent atoms and adjust atom numbering.
        """
        max_atom = self.atoms[-1][0]
        for collection in [self.dihedrals, self.angles, self.bonds]:
            for i in range(len(collection) - 1, -1, -1):
                if max(collection[i].atoms) > max_atom:
                    del collection[i]
        for collection in [self.vsites, self.exclusions, self.pairs]:
            for i in range(len(collection) - 1, -1, -1):
                if 1 in collection[i].atoms:
                    del collection[i]
                else:
                    collection[i].atoms = tuple(j - 1 for j in collection[i].atoms)
        # Remove the first atom and shift the rest.
        if self.atoms:
            del self.atoms[0]
            for i in range(len(self.atoms)):
                self.atoms[i] = (self.atoms[i][0] - 1,) + self.atoms[i][1:]
        if self.posres:
            del self.posres[-1]
