import os
import sys
# import openmm as mm
from pathlib import Path
from pdbfixer.pdbfixer import PDBFixer
from openmm.app import PDBFile


AA_CODE_CONVERTER = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR'
}


def convert_mutation_format(mutation):
    # Check if the input follows the expected format
    if len(mutation) < 3 or not mutation[0].isalpha() or not mutation[-1].isalpha() or not mutation[1:-1].isdigit():
        return "Invalid input format"
    
    # Extract components
    from_aa = mutation[0]
    to_aa = mutation[-1]
    position = mutation[1:-1]
    
    # Convert using the dictionary
    from_aa_3letter = AA_CODE_CONVERTER.get(from_aa, "UNK")
    to_aa_3letter = AA_CODE_CONVERTER.get(to_aa, "UNK")
    
    # Return the new format
    return f"{from_aa_3letter}-{position}-{to_aa_3letter}"


def prepare_aa_pdb(in_pdb, out_pdb, add_missing_atoms=False, add_hydrogens=False, variant=None):
    if variant:
        mutations = [convert_mutation_format(mutation) for mutation in variant]
    print(f"Opening {in_pdb}")
    pdb = PDBFixer(filename=in_pdb)
    if variant:
        print("Mutating residues")
        pdb.applyMutations(mutations, "A")
    print("Looking for missing residues")
    pdb.findMissingResidues()
    print("Looking for non-standard residues")
    pdb.findNonstandardResidues()
    print("Replacing non-standard residues")
    pdb.replaceNonstandardResidues()
    print("Removing heterogens")
    pdb.removeHeterogens(False)
    if add_missing_atoms:
        print("Looking for missing atoms")
        pdb.findMissingAtoms()
        print("Adding missing atoms")
        pdb.addMissingAtoms()
    if add_hydrogens:
        print("Adding missing hydrogens")
        pdb.addMissingHydrogens(7.0)
    topology = pdb.topology
    positions = pdb.positions
    print("Writing PDB")
    PDBFile.writeFile(topology, positions, open(out_pdb, 'w'))
    

def prepare_toppos(wdir):
    bdir = os.getcwd()
    os.chdir(wdir)
    input_pdb_file = 'protein_fixed.pdb'
    output_pdb_file = 'system.pdb'

    pdb = app.PDBFile(input_pdb_file)
    topology = pdb.getTopology()
    positions = pdb.getPositions()

    force_field = 'amber14-all.xml'
    water_model = 'amber14/tip3p.xml'
    water_box_padding = 16
    water_box_shape = 'cube'
    ions = {
        'positive':'Na+',
        'negative': 'Cl-'
    }
    ion_concentration = 0

    print('Loading hydrogen definitions...')
    app.Modeller.loadHydrogenDefinitions('glycam-hydrogens.xml')

    print('Creating forcefield...')
    forcefield = app.ForceField(force_field, water_model) # 

    print('Creating modeller...')
    modeller = app.Modeller(topology, positions)

    print('Adding hydrogens...')
    modeller.addHydrogens(forcefield)

    print('Adding solvent...')
    modeller.addSolvent(
        forcefield=forcefield,
        model=os.path.basename(os.path.realpath(water_model)).replace('.xml', ''),
        padding=water_box_padding*unit.angstroms.conversion_factor_to(unit.nanometers),
        boxShape=water_box_shape,
        positiveIon=ions['positive'],
        negativeIon=ions['negative'],
        ionicStrength=ion_concentration * unit.molar,
    )

    topology = modeller.topology
    positions = modeller.positions
    app.PDBFile.writeFile(topology, positions, open(output_pdb_file, 'w'))
    os.chdir(bdir)
    print(f'Topology and positions preparation completed')    


def create_system(topology):
    force_field = 'amber14-all.xml'
    water_model = 'amber14/tip3p.xml'

    forcefield = app.ForceField(force_field, water_model)

    nonbonded_method = app.PME
    nonbonded_cutoff = 12.0 * unit.angstroms.conversion_factor_to(unit.nanometers)
    constraints = app.HBonds
    rigid_water = True
    ewald_error_tolerance = 0.0005 # suggested 0.0005 i want 0.000001

    print('Creating system...')
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=constraints,
        rigidWater=rigid_water,
        ewaldErrorTolerance=ewald_error_tolerance
    )
    return system


def manage_restraints(simulation, restraint_mask, **kwargs):
    restraint_force_constant = kwargs.get(
        'restraint_force_constant', 1000.0 * unit.kilojoules_per_mole / unit.nanometer**2
    )
    action = kwargs.get('action', 'apply')

    topology = simulation.topology
    system = simulation.system

    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()

    if action == 'apply' and restraint_mask:
        print('Applying restraints...')
        force = mm.CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        force.addGlobalParameter('k', restraint_force_constant)
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')
        for atom in topology.atoms():
            if atom.residue.name not in restraint_mask:
                # print(atom, atom.index, positions[atom.index])
                force.addParticle(atom.index, positions[atom.index])
        system.addForce(force)
    elif action == 'remove':
        print('Removing restraints...')
        forces_to_remove = []
        for i, force in enumerate(system.getForces()):
            if isinstance(force, mm.CustomExternalForce):
                if force.getEnergyFunction() == 'k*periodicdistance(x, y, z, x0, y0, z0)^2':
                    forces_to_remove.append(i)
        for i in reversed(forces_to_remove):
            system.removeForce(i)
    
    for i, force in enumerate(system.getForces()):
        print(i, force)


def platform_select(platform_name, **kwargs):
    num_cores = kwargs.get('num_cores', 6)
    print('Getting platform...')
    platform = mm.Platform.getPlatformByName(platform_name)
    if platform_name == 'CPU':
        platform.setPropertyDefaultValue('Threads', str(num_cores))
    print(f'Platform: {platform}')
    return platform


def add_reporters(simulation, nsteps, output_prefix, **kwargs):
    interval = 5000
    checkpoint_interval = kwargs.get('checkpoint_interval', interval)
    dcd_interval = kwargs.get('dcd_interval', interval)
    pdb_interval = kwargs.get('pdb_interval', 50000)
    state_interval = kwargs.get('state_interval', interval)

    if nsteps <= pdb_interval:
        pdb_interval = interval

    print('Adding Reporters...')
    print('Adding CheckpointReporter...')
    checkpoint_file = f'{output_prefix}.chk'
    simulation.reporters.append(
        app.CheckpointReporter(
            checkpoint_file,
            checkpoint_interval
        )
    )
    print('Adding DCDReporter...')
    dcd_file = f'{output_prefix}.dcd'
    simulation.reporters.append(
        app.DCDReporter(
            dcd_file,
            dcd_interval
        )
    )
    print('Adding PDBReporter...')
    pdb_file = f'{output_prefix}.pdb'
    simulation.reporters.append(
        app.PDBReporter(
            pdb_file,
            pdb_interval,
            enforcePeriodicBox=True
        )
    )
    print('Adding StateDataReporter...')
    state_file = f'{output_prefix}.csv'
    simulation.reporters.append(
        app.StateDataReporter(
            state_file,
            state_interval,
            step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
            volume=True, density=True, progress=True, remainingTime=True, speed=True, elapsedTime=True,
            separator=',', systemMass=None, totalSteps=nsteps
        )
    )


def simulation_prep(**kwargs):
    platform_name = kwargs.get('platform_name', 'OpenCL')
    topology = kwargs.get('topology')
    system = kwargs.get('system')
    integrator = kwargs.get('integrator')
    input_state_file = kwargs.get('input_state_file')
    pdb = kwargs.get('pdb')
    restraint_mask = kwargs.get('restraint_mask')
    nsteps = kwargs.get('nsteps')
    output_prefix = kwargs.get('output_prefix')
    
    print('Creating simulation...')
    platform = platform_select(platform_name)
    simulation = app.Simulation(topology, system, integrator, platform)

    if Path(input_state_file).is_file():
        print(f'State file {input_state_file} exists')
        simulation.loadState(input_state_file)
    else:
        print(f'State file {input_state_file} does not exist !')
        positions = pdb.getPositions()
        simulation.context.setPositions(positions)

    manage_restraints(simulation, restraint_mask)

    add_reporters(simulation, nsteps, output_prefix)
    return simulation


def simulation_run(wdir, **kwargs):
    input_pdb_file = kwargs.get('input_pdb_file')
    input_state_file = kwargs.get('input_state_file')
    output_prefix = kwargs.get('output_prefix')
    mode = kwargs.get('mode', 'npt')
    platform_name = kwargs.get('platform_name', 'CUDA')
    temperature = kwargs.get('temperature', 300 * unit.kelvin)
    pressure = kwargs.get('pressure', 1 * unit.bar)
    nsteps = kwargs.get('nsteps', 50000)
    restraint_mask = kwargs.get('restraint_mask', '')
    start_temp = kwargs.get('start_temp', 10 * unit.kelvin)
    end_temp = kwargs.get('end_temp',temperature)

    output_state_file = f'{output_prefix}.xml'
    
    bdir = os.getcwd()
    os.chdir(wdir)

    pdb = app.PDBFile(input_pdb_file)
    topology = pdb.getTopology()
    system = create_system(topology)


    print('Creating integrator...')
    timestep = 0.002*unit.picoseconds
    integrator = mm.VerletIntegrator(timestep)

    print(f'Creating {mode} simulation...')
    simulation_parameters = {
        'platform_name': platform_name,
        'topology': topology,
        'system': system,
        'integrator': integrator,
        'input_state_file': input_state_file,
        'pdb': pdb,
        'restraint_mask': restraint_mask,
        'nsteps': nsteps,
        'output_prefix': output_prefix
    }
    simulation = simulation_prep(**simulation_parameters)

    print(f'Running {mode} simulation...')

    simulation.currentStep = 0
    simulation.minimizeEnergy(maxIterations=nsteps)

    
    simulation.saveState(f'{output_prefix}.xml')

    with open(f'{output_prefix}_last.pdb', 'w') as output:
        state = simulation.context.getState(getPositions=True)
        app.PDBFile.writeFile(topology, state.getPositions(), output, keepIds=True)
        
    os.chdir(bdir)
    print(f'{mode} simulation completed')    


def minimize_aa(wdir, platform_name='CPU'): 
    prepare_toppos(wdir)   
    
    # Solvent minimization
    print('Solvent minimization begin...')
    solvent_minimization_parameters = {
        'wdir': wdir,
        'platform_name': platform_name, 
        'input_pdb_file': 'system.pdb',
        'input_state_file': '',
        'output_prefix': 'system_minimized',
        'mode': 'min',
        'nsteps': 10000,
        'restraint_mask': ['HOH','NA','CL']
    }
    simulation_run(**solvent_minimization_parameters)
    print('Solvent minimization end...')
    bdir = os.getcwd()
    os.chdir(wdir)
    pdb = PDBFixer('system_minimized_last.pdb')
    pdb.removeHeterogens(False)
    topology = pdb.topology
    positions = pdb.positions
    app.PDBFile.writeFile(topology, positions, open(f'protein_minimized.pdb', 'w'))
    os.chdir(bdir)
    
    
    
