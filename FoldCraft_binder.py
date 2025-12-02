import argparse
import jax
import jax.numpy as jnp
import os

from colabdesign import mk_af_model

import pandas as pd
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
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
import random
from colabdesign.af.loss import get_contact_map

import matplotlib.pyplot as plt
from matplotlib import patches
from colabdesign.mpnn import mk_mpnn_model
from biopython_utils import *
import warnings

from BindCraft.functions import *
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball "./DAlphaBall.gcc" -corrections::beta_nov16 true -relax:default_repeats 1')

def parse_args():
    parser = argparse.ArgumentParser(description="Run fold-conditioned binder design")

    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the results')
    parser.add_argument('--sample', action='store_true', help='Whether to generate designs until the target number of successful designs is reached')
    parser.add_argument('--target_success', type=int, default=100, help='Target number of successful designs to generate (used only if --sample is enabled)')
    parser.add_argument('--num_designs', type=int, default=1, help='Number of design trajectories to generate (ignored if --sample is enabled)')
    parser.add_argument('--vhh', action='store_true', help='Whether to use VHH framework to construct target cmap (all binder information would be ignored in that case)')
    
    parser.add_argument('--binder_template', type=str, help='Path to the binder template PDB file (required)')
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
    parser.add_argument('--start_with', type=int, default=0, help='Start with')
    parser.add_argument('--binder_len', type=str, default='100', help='Binder len in str')
      
    return parser.parse_args()

def main():
    args = parse_args()

    #Prepare fold conditioned binder
    template_pdb = args.binder_template #template for binder
    chain_template = args.binder_chain #chain for binder

    pdb_target_path = args.target_template #template for target
    chain_id = args.target_chain #Select chain for target protein.
    binder_len = args.binder_len
    if '-' in binder_len:
    	binder_lengths = [int(i) for i in binder_len.split('-')]
    else:
    	binder_len = int(binder_len)
    	binder_lengths = False
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
    start_with = args.start_with

    model_name = 'v_48_010'

    if start_with==0:
    	try:
    	    os.system(f'mkdir {folder_name}')
    	except:
            pass  
    
    # Start to design

    # Define the fold-conditioned loss
    
    names = []
    sequences = []
    plddts = []
    ipaes = []
    iptms = []
    rg = []
    shape_c = []
    uns_hb = []
    hydrophob_per = []
    int_dG = []
    clashes = []
    rmsds = []

    # Create folders to save outputs
    if start_with == 0:
    	os.system(f'mkdir {folder_name}/traj/')
    	os.system(f'mkdir {folder_name}/mpnn/')
    	os.system(f'mkdir {folder_name}/designs/')
    	os.system(f'mkdir {folder_name}/relaxed/')

    # Generate N number of trajectories
    
    if sample:
        passed = 0
        #success_target = success_target
        i=start_with
        while passed <= success_target:
            clear_mem()
            if binder_lengths != False:
            	binder_len = random.randint(binder_lengths[0], binder_lengths[1]+1)
            i+=1
            name = f'traj_{i}' #@param {type:"string"}
            rm_aa = 'C' #@param {type:"string"}
            def rg_loss(inputs, outputs):
                positions = outputs["structure_module"]["final_atom_positions"]
                ca = positions[:,residue_constants.atom_order["CA"]]
                center = ca.mean(0)
                rg = jnp.sqrt(jnp.square(ca - center).sum(-1).mean() + 1e-8)
                rg_th = 2.38 * ca.shape[0] ** 0.365
                rg = jax.nn.elu(rg - rg_th)
                return {"rg":rg}
    
            af_model = mk_afdesign_model(protocol="binder", loss_callback=rg_loss,
                                                 use_templates=True,)
            
            af_model.prep_inputs(pdb_filename=pdb_target_path,
                                         chain=chain_id, binder_len = binder_len,
                                         hotspot=target_hotspots,
                                         rm_aa=rm_aa, #fix_pos=fixed_positions,
                                         )
            
            af_model.opt["weights"]["rg"] = .4
            af_model.opt["weights"].update({"rg":.0, "rmsd":0.0, "fape":0.0, "plddt":0.0, 'pae':0.2,
                                                    "con":1.0, "i_con":1.0, "i_pae":0.2})
            
            af_model.design_3stage(design_stages[0],design_stages[1],design_stages[2])
            #af_model.design_semigreedy(20, tries=20, models=[0,1], num_models=2)
            af_model.save_pdb(f"{folder_name}/traj/{name}.pdb", get_best=True)
            with open(f'{folder_name}/traj/{name}.pickle', 'wb') as handle:
                pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            if af_model.aux['log']['i_ptm']>0. and af_model.aux['log']['plddt']>.65:
                clear_mem()
                #Running ProteinMPNN on designed trajectory
                design_pos = range(1,binder_len) # positions to design
                interface_residues_list = list(hotspot_residues(f"{folder_name}/traj/{name}.pdb", 'B').keys())
                
                if redesign_method == 'non-interface':
                    sol_design_pos = ','.join([f'B{i}' for i in design_pos if i not in interface_residues_list])
                elif redesign_method == 'full':
                    sol_design_pos = ','.join([f'B{x}' for x in design_pos])
                else:
                    raise ValueError("Wrong redesign_method was selected. Options: 'full','non-interface'")
            
                mpnn_model = mk_mpnn_model(model_name, backbone_noise=mpnn_backbone_noise,weights=mpnn_version)
                mpnn_model.prep_inputs(pdb_filename=f"{folder_name}/traj/{name}.pdb", chain='A,B', 
                                       fix_pos=sol_design_pos, rm_aa = "C", inverse=True)
            
                samples = mpnn_model.sample_parallel(temperature=mpnn_sampling_temp, batch=mpnn_samples)
                if mpnn_save:
                    with open(f'{folder_name}/mpnn/mpnn_{name}.pickle', 'wb') as handle:
                        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                    
                #Predict Samples with AF2_ptm
                print('Predicting sequences with AF2_ptm...')    
                for num, seq in enumerate(samples['seq']):
                    af_model = mk_afdesign_model(protocol="binder", loss_callback=rg_loss,
                                                           use_templates=True,)
           
                    af_model.prep_inputs(pdb_filename=pdb_target_path,
                                                   chain=chain_id, binder_len = binder_len,
                                                   #hotspot='',
                                                   rm_aa=rm_aa, #fix_pos=fixed_positions,
                                                   )
                    af_model.set_seq(seq[-binder_len:])
                    af_model.predict(num_recycles=3, verbose=False, models=[0,1])
                    
                    print(f"predict: {name}_{num} plddt: {af_model.aux['log']['plddt']:.3f}, i_pae: {(af_model.aux['log']['i_pae']):.3f}, i_ptm: {af_model.aux['log']['i_ptm']:.3f}, rg: {af_model.aux['log']['rg']:.3f}")
                    
                    if af_model.aux['log']['i_pae']<0.35 and af_model.aux['log']['plddt']>.8 and af_model.aux['log']['i_ptm']>0.5:
                        af_model.save_pdb(f"{folder_name}/designs/{name}_{num}.pdb", get_best=False)
                        with open(f'{folder_name}/designs/{name}_{num}.pickle', 'wb') as handle:
                            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

                        mpnn_design_relaxed = f'{folder_name}/relaxed/{name}_{num}.pdb'
                        mpnn_design_pdb = f'{folder_name}/designs/{name}_{num}.pdb'
                        pr_relax(mpnn_design_pdb, mpnn_design_relaxed)
                        binder_chain = 'B'
                        num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)
                        mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_design_relaxed, binder_chain)
                        
                        if num_clashes_mpnn_relaxed==0 and mpnn_interface_scores['interface_sc']>0.55 and mpnn_interface_scores['interface_delta_unsat_hbonds'] < 3 and mpnn_interface_scores['surface_hydrophobicity'] <= 0.35:
                            clashes.append(num_clashes_mpnn_relaxed)
                            shape_c.append(mpnn_interface_scores['interface_sc'])
                            uns_hb.append(mpnn_interface_scores['interface_delta_unsat_hbonds'])
                            hydrophob_per.append(mpnn_interface_scores['surface_hydrophobicity'])
                            int_dG.append(mpnn_interface_scores['interface_dG'])
                            
                            names.append(f'{name}_{num}')
                            sequences.append(seq[-binder_len:])
                            plddts.append(af_model.aux['log']['plddt'])
                            ipaes.append(af_model.aux['log']['i_pae'])
                            iptms.append(af_model.aux['log']['i_ptm'])
                            rg.append(af_model.aux['log']['rg'])
                            
                            af_model = mk_afdesign_model(protocol="fixbb", use_templates=False)
                            af_model.prep_inputs(pdb_filename=f'{folder_name}/designs/{name}_{num}.pdb', chain='B',
                                                 length=binder_len, ignore_missing=False)
                            af_model.set_seq(seq[-binder_len:])
                            af_model.predict(num_recycles=3, verbose=False)
                            
                            rmsds.append(af_model.aux['log']['rmsd'])
                            df = pd.DataFrame({'name':names,
                               'sequence':sequences,
                               'plddt':plddts,
                               'ipae':ipaes,
                               'iptm':iptms,
                               'rg_loss':rg,
                               'shape_c': shape_c,
                                'uns_hb' : uns_hb,
                                'hydrophob_per' : hydrophob_per,
                                'int_dG' : int_dG,
                                'clashes' : clashes,
                                'rmsds': rmsds,})
    
                            df.to_csv(f"{folder_name}/results_pyrosetta.csv")
                            passed+=1
                        else:
                            
                            os.system(f'rm {mpnn_design_relaxed}')
                        
                        
    
    df = pd.DataFrame({'name':names,
                               'sequence':sequences,
                               'plddt':plddts,
                               'ipae':ipaes,
                               'iptm':iptms,
                               'rg_loss':rg,
                               'shape_c': shape_c,
                                'uns_hb' : uns_hb,
                                'hydrophob_per' : hydrophob_per,
                                'int_dG' : int_dG,
                                'clashes' : clashes,
                                'rmsds': rmsds,})
    
    df.to_csv(f"{folder_name}/results.csv")

if __name__ == '__main__':
    main()
