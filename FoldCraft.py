import argparse
import jax
import jax.numpy as jnp
import os

from colabdesign import mk_af_model

import pandas as pd
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import numpy as np
import pickle
from tqdm.notebook import tqdm
import glob

from colabdesign.af.alphafold.common import protein
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import residue_constants

import os
from colabdesign import mk_afdesign_model, clear_mem
import numpy as np

from colabdesign.af.loss import get_contact_map

import matplotlib.pyplot as plt
from matplotlib import patches
from colabdesign.mpnn import mk_mpnn_model
from biopython_utils import *
from cmap_utils import assemble_fold_conditioned_cmap, binarize_cmap
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Run fold-conditioned binder design")

    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the results')
    parser.add_argument('--sample', action='store_true', help='Whether to generate designs until the target number of successful designs is reached')
    parser.add_argument('--target_success', type=int, default=100, help='Target number of successful designs to generate (used only if --sample is enabled)')
    parser.add_argument('--num_designs', type=int, default=1, help='Number of design trajectories to generate (ignored if --sample is enabled)')
    parser.add_argument('--vhh', action='store_true', help='Whether to use VHH framework to construct target cmap (all binder information would be ignored in that case)')
    
    parser.add_argument('--binder_template', type=str, default='', help='Path to the binder template PDB file (required unless --vhh is set)')
    parser.add_argument('--target_template', type=str, required=True, help='Path to the target template PDB file (required)')
    parser.add_argument('--target_hotspots', type=str, required=True, help='''Residue ranges for target hotspots, e.g., "14-30,80-81,90-102" (required)''')
    parser.add_argument('--binder_hotspots', type=str, default='', help='Residue ranges for binder, e.g. "14-30,80-81,90-102"') 
    parser.add_argument('--binder_mask', type=str, default='', help='Residue ranges in the binder to mask (ignored during loss computation), e.g. "14-30"') 
 
    parser.add_argument('--binder_chain', type=str, default='A', help='Binder template chain (default = A)') 
    parser.add_argument('--target_chain', type=str, default='A', help='Target template chain (default = A)') 

    parser.add_argument('--design_stages', type=str, default='100,100,20', help="Number of each design stages in 3stage_design (default: 100,100,20)")
    
    parser.add_argument('--mpnn_weight', type=str, choices=['soluble', 'original'], default='soluble', help="PoteinMPNN weights to use ('soluble', 'original')")
    parser.add_argument('--redesign_method', type=str, choices=['full', 'non-interface'], default='non-interface', help="ProteinMPNN redesign strategy: 'full' or 'non-interface' (default: 'non-interface')") 
    parser.add_argument('--mpnn_samples', type=int, default=5, help="Number of sequences to sample with ProteinMPNN (default: 5)")
    parser.add_argument('--mpnn_backbone_noise', type=float, default=0.0, help="Backbone noise during sampling (default: 0.0)")
    parser.add_argument('--mpnn_sampling_temp', type=float, default=0.1, help="Sampling temperature for amino acids 0.0-1.0 (default: 0.1)")
    parser.add_argument('--mpnn_save', action='store_true', help='Whether to save MPNN sampled sequences')
      
    return parser.parse_args()

def main():
    args = parse_args()

    #Prepare fold conditioned binder
    template_pdb = args.binder_template #template for binder
    binder_template = template_pdb.split('/')[-1].split('.')[0]
    chain_template = args.binder_chain #chain for binder
    vhh = args.vhh

    if not vhh and not template_pdb:
        raise SystemExit("Error: --binder_template is required unless --vhh is set.")

    pdb_target_path = args.target_template #template for target
    chain_id = args.target_chain #Select chain for target protein.
    
    target_hotspots = args.target_hotspots #Choose hotspots on target protein
    binder_hotspots = args.binder_hotspots #Optional: Choose hotspots on binder protein
    
    binder_mask = args.binder_mask
    
    folder_name = args.output_folder #name for output folder
    
    redesign_method = args.redesign_method
    mpnn_version = args.mpnn_weight

    sample = args.sample
    success_target = args.target_success
    
    num_designs = args.num_designs
    mpnn_samples = args.mpnn_samples

    design_stages = [int(x) for x in args.design_stages.split(',')]
    mpnn_backbone_noise = args.mpnn_backbone_noise
    mpnn_sampling_temp = args.mpnn_sampling_temp

    mpnn_save = args.mpnn_save

    model_name = 'v_48_010'
    os.makedirs(folder_name, exist_ok=True)

    if vhh:
        # VHH path: binder cmap comes from the fixed VHH framework, the binder
        # length is the 127-residue VHH scaffold, and the CDR (binder) hotspots
        # are fixed -- any user-supplied binder_hotspots/binder_mask is ignored,
        # matching the original behavior.
        # Anchor the bundled VHH framework cmap to the script's directory so
        # --vhh runs work from any CWD (SLURM job dirs, installed invocations);
        # the previous CWD-relative 'framework/vhh.npy' raised FileNotFoundError
        # whenever the process was not launched from the repo root.
        vhh_framework = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'framework', 'vhh.npy')
        load_np = np.load(vhh_framework)
        af_model = mk_afdesign_model(protocol="fixbb", use_templates=True)
        af_model.prep_inputs(pdb_filename=pdb_target_path,
                             ignore_missing=False,
                             chain = chain_id,)

        target_len = af_model._len
        binder_len = 127
        binder_hotspots = '26-35,55-59,102-116'
        fc_cmap = assemble_fold_conditioned_cmap(load_np, target_len, binder_len,
                                                 target_hotspots, binder_hotspots)
    else:
        # Count the binder template's *polymer* residues for the gap-numbering
        # guard below. is_aa() excludes waters/ions/ligands but keeps modified
        # residues (e.g. MSE) that colabdesign promotes -- so the count matches the
        # model's view. The sequence itself is taken from the model after
        # prep_inputs (query_chain, below), NOT from seq1() over raw resnames: the
        # latter let HETATM/water inflate the length (waters -> 'X') or frame-shift
        # it (2-char ion names like ZN/NA), which spuriously aborted valid
        # crystallographic templates at the guard and, more rarely, silently
        # corrupted the conditioned cmap.
        structure = PDBParser(QUIET=True).get_structure(binder_template, template_pdb)
        n_polymer = sum(1 for residue in structure[0][chain_template] if is_aa(residue))

        af_binder = mk_afdesign_model(protocol="fixbb", use_templates=True)
        af_binder.prep_inputs(pdb_filename=template_pdb,
                             ignore_missing=False,
                             chain = chain_template,
                             rm_template_seq=False,
                             rm_template_sc=False,)

        # A polymer-count vs model-span mismatch means the binder template has gaps
        # in its residue numbering (non-contiguous resSeq), which otherwise fails
        # deep inside the model with an opaque broadcasting error.
        if n_polymer != af_binder._len:
            raise SystemExit(
                f"Error: binder template '{template_pdb}' (chain '{chain_template}') "
                f"has {n_polymer} polymer residues but the model spans "
                f"{af_binder._len} positions. This usually means the PDB has gaps in "
                "its residue numbering (missing resSeq entries). Renumber the binder "
                "template contiguously starting from 1 (and update --binder_hotspots "
                "accordingly) before running."
            )

        # Native binder sequence exactly as colabdesign sees it: aligned to _len,
        # HETATM/water-free, and MODRES-correct (MSE -> M, not 'X').
        query_chain = ''.join(
            residue_constants.restypes[a] if a < residue_constants.restype_num else 'X'
            for a in af_binder._wt_aatype)
        af_binder.set_seq(query_chain)
        af_binder.predict(num_recycles=3, verbose=False)
        print(f"CMAP of {binder_template} (monomer plddt: {af_binder.aux['log']['plddt']:.3f})")
        #plt.imshow(af_model.aux['cmap'])


        warnings.filterwarnings("ignore")

        #Prepare target protein structure

        af_model = mk_afdesign_model(protocol="fixbb", use_templates=True)
        af_model.prep_inputs(pdb_filename=pdb_target_path,
                             ignore_missing=False,
                             chain = chain_id,)

        target_len = af_model._len
        binder_len = af_binder._len

        # binder cmap from the AlphaFold2 prediction of the binder monomer
        load_np = af_binder.aux['cmap']
        fc_cmap = assemble_fold_conditioned_cmap(load_np, target_len, binder_len,
                                                 target_hotspots, binder_hotspots,
                                                 binder_mask)

    from matplotlib import patches
    fig, ax = plt.subplots()
    plt.imshow(fc_cmap)
    rect = patches.Rectangle((0, 0), target_len, target_len, linewidth=2, edgecolor='b', facecolor='none')
    rect2 = patches.Rectangle((target_len, target_len), binder_len, binder_len, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(rect2)
    plt.savefig(f'{folder_name}/fold_cond_cmap.png')

    np.save(f'{folder_name}/fold_cond_cmap.npy',fc_cmap)

    np.save(f'{folder_name}/fold_cond_cmap_mask.npy', binarize_cmap(fc_cmap))

    # The fold-conditioned cmap and its mask are constant for the whole run, so
    # keep them in memory instead of re-reading the .npy files from disk on every
    # model build (previously np.load'd once per trajectory and once per MPNN
    # sample). Each model is handed a fresh copy to preserve the previous
    # per-iteration isolation; the loss callback only reads these arrays.
    cond_cmap = fc_cmap
    cond_cmap_mask = binarize_cmap(fc_cmap)

    # Start to design

    # Define the fold-conditioned loss
    def custom_pre_callback(inputs, aux, opt, key):
        # save fold-conditioned cmap for binder-target 
        # complex as custom parameter inside of af_model 
        # input parameters
        
        aux["cond_cmap"] = opt["cond_cmap"]
        aux["cond_cmap_mask"] = opt["cond_cmap_mask"]
    
    def cmap_loss_binder(inputs, outputs, opt):
        # define cmap similarity loss
        
        # load fold-conditioned cmap and masked cmap from
        # custom parameters defined in custom_pre_callback
        conditioned_array = opt['cond_cmap']
        conditioned_mask = opt['cond_cmap_mask']
        binder_len = inputs['seq']['input'].shape[1]

        # calculate the cmaps for predicted structure of 
        # binder-target complex during each step with different 
        # cutoffs for intra- and inter- chain contacts
        i_cmap = get_contact_map(outputs, inputs["opt"]["i_con"]["cutoff"])
        cmap = get_contact_map(outputs, inputs["opt"]["con"]["cutoff"])

        # mask calculated cmap to restrict the loss for conditioned areas only
        i_cmap = i_cmap.at[-binder_len:,-binder_len:].set(cmap[-binder_len:,-binder_len:])
        out_cmap_conditioned = i_cmap * conditioned_mask

        # calculate the RMSE between predicted and fold-conditioned cmaps
        cmap_loss_binder = jnp.sqrt(jnp.square(out_cmap_conditioned - conditioned_array).sum(-1).mean())
    
        return {"cmap_loss_binder":cmap_loss_binder}
    
    names = []
    sequences = []
    plddts = []
    ipaes = []
    iptms = []
    cmap_loss = []

    # Persist results incrementally so an OOM/RunTimeError/preemption mid-run keeps
    # every design recorded so far, instead of losing the whole table (it was
    # previously written only once, after both loops). Progress streams to
    # results.csv.partial after each accepted design; results.csv itself is created
    # only by the finalizing call at the very end (atomic os.replace via
    # write_atomic). This keeps "results.csv exists == chunk complete" true, which
    # baseline/scheduler.py relies on for resume/merge -- a preempted chunk leaves
    # results.csv.partial (recoverable) but no results.csv, so the scheduler
    # re-runs it instead of skipping/merging it as done.
    results_csv = f"{folder_name}/results.csv"
    def write_results(finalize=False):
        df = pd.DataFrame({'name':names,
                           'sequence':sequences,
                           'plddt':plddts,
                           'ipae':ipaes,
                           'iptm':iptms,
                           'cmap_loss':cmap_loss})
        write_atomic(results_csv, df.to_csv, finalize=finalize)

    # Create folders to save outputs
    os.makedirs(f'{folder_name}/traj/', exist_ok=True)
    os.makedirs(f'{folder_name}/mpnn/', exist_ok=True)
    os.makedirs(f'{folder_name}/designs/', exist_ok=True)

    # Build ProteinMPNN once: model_name/backbone_noise/weights are constant for
    # the whole run, but mk_mpnn_model joblib-loads the weights and rebuilds its
    # jitted score/sample functions on every construction. Constructing it per
    # trajectory (as before) repeated that work each iteration. Only prep_inputs
    # (which writes self._inputs, not the model) varies per trajectory, so build
    # here and re-prep inside the loop. The sampler RNG is seeded at construction
    # with seed=None (non-deterministic), so a single shared key stream is
    # statistically identical to a fresh model per trajectory -- no distribution
    # change, just no reload.
    mpnn_model = mk_mpnn_model(model_name, backbone_noise=mpnn_backbone_noise,
                               weights=mpnn_version)

    # Generate N number of trajectories
    if sample == False:
        
        for i in range(num_designs):
            i+=1
            clear_mem() # clearing memory at each step helps to avoid RunTimeError
            # clear_mem() deletes every live JAX device buffer, including the hoisted
            # mpnn_model's RNG key (its params are host-side numpy and survive), which
            # would break sample_parallel below. Re-seed it: set_seed(None) gives a
            # fresh key per trajectory, matching the original per-trajectory build.
            mpnn_model.set_seed(None)
            name = f'traj_{i}'
            rm_aa = 'C'
            
            af_model = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                                                 use_templates=True,)

            # load the fold-conditioned cmaps inside of af_model
            af_model.opt['cond_cmap'] = cond_cmap.copy()
            af_model.opt['cond_cmap_mask'] = cond_cmap_mask.copy()
            
            af_model.prep_inputs(pdb_filename=pdb_target_path,
                                         chain=chain_id, binder_len = binder_len,
                                         hotspot=target_hotspots,
                                         rm_aa=rm_aa, #fix_pos=fixed_positions,
                                         )

            # use only cmap similarity loss during the design - it should also improve all other metrics too
            af_model.opt["weights"]["cmap_loss_binder"] = 1.
            af_model.opt["weights"].update({"cmap_loss_binder":1.0, "rmsd":0.0, "fape":0.0, "plddt":0.0,
                                                    "con":0.0, "i_con":0.0, "i_pae":0.})
            
            af_model.design_3stage(design_stages[0],design_stages[1],design_stages[2])
            af_model.save_pdb(f"{folder_name}/traj/{name}.pdb", get_best=False)

            with open(f'{folder_name}/traj/{name}.pickle', 'wb') as handle:
                pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            #Running ProteinMPNN on designed trajectory
            design_pos = range(1,binder_len) # positions to design
            interface_residues_list = list(hotspot_residues(f"{folder_name}/traj/{name}.pdb", 'B').keys())
        
            if redesign_method == 'non-interface':
                sol_design_pos = ','.join([f'B{i}' for i in design_pos if i not in interface_residues_list])
            elif redesign_method == 'full':
                sol_design_pos = ','.join([f'B{x}' for x in design_pos])
            else:
                raise ValueError("Wrong redesign_method was selected. Options: 'full','non-interface'")
        
            mpnn_model.prep_inputs(pdb_filename=f"{folder_name}/traj/{name}.pdb", chain='A,B',
                                   fix_pos=sol_design_pos, rm_aa = "C", inverse=True)
        
            samples = mpnn_model.sample_parallel(temperature=mpnn_sampling_temp, batch=mpnn_samples)
            
            # save sequences in pickle file if --mpnn_save enabled
            if mpnn_save:
                with open(f'{folder_name}/mpnn/mpnn_{name}.pickle', 'wb') as handle:
                    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                
            #Predict Samples with AF2_ptm
            print('Predicting sequences with AF2_ptm...')
            # Build the AF2-ptm prediction model once per trajectory and reuse it
            # across the mpnn_samples sequences (only set_seq changes). Previously a
            # fresh model + prep_inputs was constructed for every sample, re-running
            # template featurization and forcing a fresh XLA compile each time.
            af_model = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                                                   use_templates=True,)
            af_model.opt['cond_cmap'] = cond_cmap.copy()
            af_model.opt['cond_cmap_mask'] = cond_cmap_mask.copy()
            af_model.prep_inputs(pdb_filename=pdb_target_path,
                                           chain=chain_id, binder_len = binder_len,
                                           hotspot=target_hotspots,
                                           rm_aa=rm_aa, #fix_pos=fixed_positions,
                                           )
            for num, seq in enumerate(samples['seq']):
                af_model.set_seq(seq[-binder_len:])
                af_model.predict(num_recycles=3, verbose=False, models=["model_1_ptm","model_2_ptm"])
                print(f"predict: {name}_{num} plddt: {af_model.aux['log']['plddt']:.3f}, i_pae: {(af_model.aux['log']['i_pae']):.3f}, i_ptm: {af_model.aux['log']['i_ptm']:.3f}, cmap_loss: {af_model.aux['log']['cmap_loss_binder']:.3f}")
                af_model.save_pdb(f"{folder_name}/designs/{name}_{num}.pdb", get_best=False)
                with open(f'{folder_name}/designs/{name}_{num}.pickle', 'wb') as handle:
                    pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                names.append(f'{name}_{num}')
                sequences.append(seq)
                plddts.append(af_model.aux['log']['plddt'])
                ipaes.append(af_model.aux['log']['i_pae'])
                iptms.append(af_model.aux['log']['i_ptm'])
                cmap_loss.append(af_model.aux['log']['cmap_loss_binder'])
                write_results()   # checkpoint after each design

    else:
        passed = 0
        #success_target = success_target
        i=0
        # Stop *starting* trajectories once the target is reached (`<`, not `<=`,
        # which overshot by one). The inner MPNN-batch loop is *also* capped, via
        # iter_until_target below -- otherwise the final batch kept saving every
        # passing sample past the target (up to mpnn_samples-1 extra), since the
        # outer guard only fires between trajectories. Together they make the
        # accepted-design count exactly --target_success.
        while passed < success_target:
            clear_mem()
            mpnn_model.set_seed(None)  # clear_mem() deletes the hoisted mpnn_model's
                                       # RNG key; re-seed it (see the --num_designs branch)
            i+=1
            name = f'traj_{i}' #@param {type:"string"}
            rm_aa = 'C' #@param {type:"string"}
            
            af_model = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                                                 use_templates=True,)
            
            af_model.opt['cond_cmap'] = cond_cmap.copy()
            af_model.opt['cond_cmap_mask'] = cond_cmap_mask.copy()
            
            af_model.prep_inputs(pdb_filename=pdb_target_path,
                                         chain=chain_id, binder_len = binder_len,
                                         hotspot=target_hotspots,
                                         rm_aa=rm_aa, #fix_pos=fixed_positions,
                                         )
            
            af_model.opt["weights"]["cmap_loss_binder"] = 1.
            af_model.opt["weights"].update({"cmap_loss_binder":1.0, "rmsd":0.0, "fape":0.0, "plddt":0.0,
                                                    "con":0.0, "i_con":0.0, "i_pae":0.})
            
            af_model.design_3stage(design_stages[0],design_stages[1],design_stages[2])
            af_model.save_pdb(f"{folder_name}/traj/{name}.pdb", get_best=False)
            with open(f'{folder_name}/traj/{name}.pickle', 'wb') as handle:
                pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            if af_model.aux['log']['i_pae']<0.4 and af_model.aux['log']['plddt']>.7:
                #Running ProteinMPNN on designed trajectory
                design_pos = range(1,binder_len) # positions to design
                interface_residues_list = list(hotspot_residues(f"{folder_name}/traj/{name}.pdb", 'B').keys())
            
                if redesign_method == 'non-interface':
                    sol_design_pos = ','.join([f'B{i}' for i in design_pos if i not in interface_residues_list])
                elif redesign_method == 'full':
                    sol_design_pos = ','.join([f'B{x}' for x in design_pos])
                else:
                    raise ValueError("Wrong redesign_method was selected. Options: 'full','non-interface'")
            
                mpnn_model.prep_inputs(pdb_filename=f"{folder_name}/traj/{name}.pdb", chain='A,B',
                                       fix_pos=sol_design_pos, rm_aa = "C", inverse=True)
            
                samples = mpnn_model.sample_parallel(temperature=mpnn_sampling_temp, batch=mpnn_samples)
                if mpnn_save:
                    with open(f'{folder_name}/mpnn/mpnn_{name}.pickle', 'wb') as handle:
                        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                    
                #Predict Samples with AF2_ptm
                print('Predicting sequences with AF2_ptm...')
                # Build the AF2-ptm prediction model once per trajectory and reuse it
                # across the batch (only set_seq changes), as in the non-sample path.
                af_model = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                                                       use_templates=True,)
                af_model.opt['cond_cmap'] = cond_cmap.copy()
                af_model.opt['cond_cmap_mask'] = cond_cmap_mask.copy()
                af_model.prep_inputs(pdb_filename=pdb_target_path,
                                               chain=chain_id, binder_len = binder_len,
                                               hotspot=target_hotspots,
                                               rm_aa=rm_aa, #fix_pos=fixed_positions,
                                               )
                # Cap the batch at the remaining global budget so it can't push
                # `passed` past success_target (see iter_until_target); lambda reads
                # the live `passed`, which is incremented on each accepted design.
                for num, seq in iter_until_target(samples['seq'], lambda: passed, success_target):
                    af_model.set_seq(seq[-binder_len:])
                    af_model.predict(num_recycles=3, verbose=False, models=["model_1_ptm","model_2_ptm"])
                    print(f"predict: {name}_{num} plddt: {af_model.aux['log']['plddt']:.3f}, i_pae: {(af_model.aux['log']['i_pae']):.3f}, i_ptm: {af_model.aux['log']['i_ptm']:.3f}, cmap_loss: {af_model.aux['log']['cmap_loss_binder']:.3f}")
                    if af_model.aux['log']['i_pae']<0.35 and af_model.aux['log']['plddt']>.8 and af_model.aux['log']['i_ptm']>0.5:
                        af_model.save_pdb(f"{folder_name}/designs/{name}_{num}.pdb", get_best=False)
                        with open(f'{folder_name}/designs/{name}_{num}.pickle', 'wb') as handle:
                            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        names.append(f'{name}_{num}')
                        sequences.append(seq)
                        plddts.append(af_model.aux['log']['plddt'])
                        ipaes.append(af_model.aux['log']['i_pae'])
                        iptms.append(af_model.aux['log']['i_ptm'])
                        cmap_loss.append(af_model.aux['log']['cmap_loss_binder'])
                        passed+=1
                        write_results()   # checkpoint after each design
    
    # Final flush + atomic promote: results.csv now exists, signalling the chunk
    # completed (also writes an empty table if no designs passed, as before).
    write_results(finalize=True)

if __name__ == '__main__':
    main()