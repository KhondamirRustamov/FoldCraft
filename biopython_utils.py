import os
import math
import numpy as np
from scipy.spatial import cKDTree
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, Selection, Superimposer
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa

def set_range(hotspots_input):
    new_h = [x for x in hotspots_input.split(',')]
    h_range = []
    for i in new_h:
        if '-' in i:
            # inclusive of both endpoints: "39-45" -> 39..45 (was exclusive of
            # the upper bound, which silently dropped the last residue of every
            # hotspot/mask window).
            lo, hi = i.split('-')
            h_range += [x for x in range(int(lo), int(hi) + 1)]
        else:
            h_range.append(int(i))
    return h_range


def iter_until_target(items, current_count, target):
    """Enumerate ``items`` as ``(index, item)`` but stop *before* yielding once
    ``current_count()`` has reached ``target``.

    Caps a batch loop against a GLOBAL running total so a single batch cannot
    overshoot it. ``current_count`` is a zero-arg callable read at the top of
    every iteration, so it reflects acceptances the caller makes *inside* the
    loop body (e.g. ``current_count=lambda: passed`` where the body does
    ``passed += 1`` for each accepted item). Items are yielded lazily, so any
    per-item work past the target (here: an AF2 prediction per sample) is never
    started.

    Used by FoldCraft's ``--sample`` loop: ProteinMPNN emits a batch of
    ``mpnn_samples`` sequences per trajectory and every sequence clearing the
    success gate is saved; without this cap the final batch kept saving past
    ``--target_success`` (overshooting by up to ``mpnn_samples - 1``). The outer
    ``while passed < target`` only stops *starting* new trajectories, not the
    in-flight batch -- this is what makes the success count exact.
    """
    it = iter(items)
    index = 0
    while current_count() < target:
        try:
            item = next(it)
        except StopIteration:
            return
        yield index, item
        index += 1


def write_atomic(final_path, write_fn, finalize=False):
    """Write output via a temporary ``.partial`` file, promoting it to
    ``final_path`` only on ``finalize`` (an atomic ``os.replace``) -- so
    ``final_path`` exists only once it is fully written.

    ``write_fn(path)`` performs the actual write (e.g. ``df.to_csv``). Every call
    (re)writes ``final_path + '.partial'``; with ``finalize=True`` that partial is
    atomically renamed onto ``final_path``. A crash before the finalizing call
    leaves only the ``.partial`` (every row written so far, for recovery) and
    never a half-written ``final_path``.

    This matters when a consumer treats the file's *existence* as "complete":
    baseline/scheduler.py keys chunk resume/merge on ``results.csv`` existing, so
    streaming progress straight into ``results.csv`` would make a preempted chunk
    look finished. Streaming into ``results.csv.partial`` and promoting once, at
    the end, keeps "``results.csv`` exists == chunk done" true. Returns the
    partial path.
    """
    partial = final_path + ".partial"
    write_fn(partial)
    if finalize:
        os.replace(partial, final_path)
    return partial

# analyze sequence composition of design
def validate_design_sequence(sequence, num_clashes, advanced_settings):
    note_array = []

    # Check if protein contains clashes after relaxation
    if num_clashes > 0:
        note_array.append('Relaxed structure contains clashes.')

    # Check if the sequence contains disallowed amino acids
    if advanced_settings["omit_AAs"]:
        restricted_AAs = advanced_settings["omit_AAs"].split(',')
        for restricted_AA in restricted_AAs:
            if restricted_AA in sequence:
                note_array.append('Contains: '+restricted_AA+'!')

    # Analyze the protein
    analysis = ProteinAnalysis(sequence)

    # Calculate the reduced extinction coefficient per 1% solution
    extinction_coefficient_reduced = analysis.molar_extinction_coefficient()[0]
    molecular_weight = round(analysis.molecular_weight() / 1000, 2)
    extinction_coefficient_reduced_1 = round(extinction_coefficient_reduced / molecular_weight * 0.01, 2)

    # Check if the absorption is high enough
    if extinction_coefficient_reduced_1 <= 2:
        note_array.append(f'Absorption value is {extinction_coefficient_reduced_1}, consider adding tryptophane to design.')

    # Join the notes into a single string
    notes = ' '.join(note_array)

    return notes

# temporary function, calculate RMSD of input PDB and trajectory target
def target_pdb_rmsd(trajectory_pdb, starting_pdb, chain_ids_string):
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure('trajectory', trajectory_pdb)
    structure_starting = parser.get_structure('starting', starting_pdb)
    
    # Extract chain A from trajectory_pdb
    chain_trajectory = structure_trajectory[0]['A']
    
    # Extract the specified chains from starting_pdb
    chain_ids = chain_ids_string.split(',')
    residues_starting = []
    for chain_id in chain_ids:
        chain_id = chain_id.strip()
        chain = structure_starting[0][chain_id]
        for residue in chain:
            if is_aa(residue, standard=True):
                residues_starting.append(residue)
    
    # Extract residues from chain A in trajectory_pdb
    residues_trajectory = [residue for residue in chain_trajectory if is_aa(residue, standard=True)]
    
    # Ensure that both structures have the same number of residues
    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]
    
    # Collect CA atoms from the two sets of residues
    atoms_starting = [residue['CA'] for residue in residues_starting if 'CA' in residue]
    atoms_trajectory = [residue['CA'] for residue in residues_trajectory if 'CA' in residue]
    
    # Calculate RMSD using structural alignment
    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    rmsd = sup.rms
    
    return round(rmsd, 2)

# detect C alpha clashes for deformed trajectories
def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms = []
    atom_info = []  # Detailed atom info for debugging and processing

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':  # Skip hydrogen atoms
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes, only consider clashes between different chains
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)

three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues