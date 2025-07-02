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

from colabdesign.af.loss import get_contact_map

import matplotlib.pyplot as plt
from matplotlib import patches
from colabdesign.mpnn import mk_mpnn_model
from biopython_utils import *
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Run fold-conditioned binder design")

    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the results')
    parser.add_argument('--sample', action='store_true', help='Whether to generate designs until the target number of successful designs is reached')
    parser.add_argument('--target_success', type=int, default=100, help='Target number of successful designs to generate (used only if --sample is enabled)')
    parser.add_argument('--num_designs', type=int, default=1, help='Number of design trajectories to generate (ignored if --sample is enabled)')
    parser.add_argument('--vhh', action='store_true', help='Whether to use VHH framework to construct target cmap (all binder information would be ignored in that case)')
    
    parser.add_argument('--binder_template', type=str, required=True, help='Path to the binder template PDB file (required)')
    parser.add_argument('--target_template', type=str, required=True, help='Path to the target template PDB file (required)')
    parser.add_argument('--target_hotspots', type=str, required=True, help='''Residue ranges for target hotspots, e.g., "14-30,80-81,90-102" (required)''')
    parser.add_argument('--binder_hotspots', type=str, default='', help='Residue ranges for binder, e.g. "14-30,80-81,90-102"') 
    parser.add_argument('--binder_mask', type=str, default='', help='Residue ranges in the binder to mask (ignored during loss computation), e.g. "14-30"') 
 
    parser.add_argument('--binder_chain', type=str, default='A', help='Binder template chain (default = A)') 
    parser.add_argument('--target_chain', type=str, default='A', help='Target template chain (default = A)') 

    parser.add_argument('--design_stages', type=str, default='100,100,20', help="Number of each design stages in 3stage_design (default: 100,100,20)")
    
    parser.add_argument('--mpnn_weight', type=str, choices=['soluble', 'stable'], default='soluble', help="PoteinMPNN weights to use ('soluble', 'original')")
    parser.add_argument('--redesign_method', type=str, choices=['full', 'non-interface'], default='non-interface', help="ProteinMPNN redesign strategy: 'full' or 'non-interface' (default: 'non-interface')") 
    parser.add_argument('--mpnn_samples', type=int, default=5, help="Number of sequences to sample with ProteinMPNN (default: 5)")
    parser.add_argument('--mpnn_backbone_noise', type=int, default=0.0, help="Backbone noise during sampling (default: 0.0)")
    parser.add_argument('--mpnn_sampling_temp', type=int, default=0.1, help="Sampling temperature for amino acids 0.0-1.0 (default: 0.1)")
    parser.add_argument('--mpnn_save', action='store_true', help='Whether to save MPNN sampled sequences')
      
    return parser.parse_args()

def main():
    args = parse_args()

    #Prepare fold conditioned binder
    template_pdb = args.binder_template #template for binder
    binder_template = template_pdb.split('/')[-1].split('.')[0]
    chain_template = args.binder_chain #chain for binder
    vhh = args.vhh

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
    try:
        os.system(f'mkdir {folder_name}')
    except:
        pass
        
    if vhh:
        load_np = np.load(f'framework/vhh.npy')
        target_hotspots_np = np.array(set_range(target_hotspots))
        af_model = mk_afdesign_model(protocol="fixbb", use_templates=True)
        af_model.prep_inputs(pdb_filename=pdb_target_path,
                             ignore_missing=False,
                             chain = chain_id,)
        
        target_len = af_model._len
        binder_len = 127
        fc_cmap = np.zeros((target_len+binder_len, target_len+binder_len))
        
        binder_hotspots = '26-35,55-59,102-116'
        cdr_range = np.array(set_range(binder_hotspots))+target_len
        
        fc_cmap[-binder_len:,-binder_len:] = load_np
        
        for i in target_hotspots_np:
            for x in cdr_range:
                fc_cmap[x-1,i-1] = 1.
                fc_cmap[i-1,x-1] = 1.
    else:
        pdbparser = PDBParser()
        
        structure = pdbparser.get_structure(binder_template, template_pdb)
        chains = {chain.id:seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}
        
        query_chain = chains[chain_template]
        
        af_binder = mk_afdesign_model(protocol="fixbb", use_templates=True)
        af_binder.prep_inputs(pdb_filename=template_pdb,
                             ignore_missing=False,
                             chain = chain_template,
                             rm_template_seq=False,
                             rm_template_sc=False,)
        
        #name='9had'
        af_binder.set_seq(query_chain[:af_binder._len])
        af_binder.predict(num_recycles=3, verbose=False)
        print(f"CMAP of {binder_template} (monomer plddt: {af_binder.aux['log']['plddt']:.3f})")
        #plt.imshow(af_model.aux['cmap'])
        
        
        warnings.filterwarnings("ignore")
        
        #Prepare target protein structure
        
        target_hotspots_np = np.array(set_range(target_hotspots))
        
        af_model = mk_afdesign_model(protocol="fixbb", use_templates=True)
        af_model.prep_inputs(pdb_filename=pdb_target_path,
                             ignore_missing=False,
                             chain = chain_id,)
        
        target_len = af_model._len
        binder_len = af_binder._len
        
        load_np = af_binder.aux['cmap']
        
        if binder_mask != '':
            binder_mask = set_range(binder_mask)
            for i in binder_mask:
                load_np[i,:] = 0.
                load_np[:,i] = 0.
        
        fc_cmap = np.zeros((target_len+binder_len, target_len+binder_len))
        
        if binder_hotspots == '':
            cdr_range = np.array([range(0,binder_len)])+target_len
        else:
            cdr_range = np.array(set_range(binder_hotspots))+target_len
        
        fc_cmap[-binder_len:,-binder_len:] = load_np
        
        for i in target_hotspots_np:
            for x in cdr_range:
                fc_cmap[x-1,i-1] = 1.
                fc_cmap[i-1,x-1] = 1.
    
    from matplotlib import patches
    fig, ax = plt.subplots()
    plt.imshow(fc_cmap)
    rect = patches.Rectangle((0, 0), target_len, target_len, linewidth=2, edgecolor='b', facecolor='none')
    rect2 = patches.Rectangle((target_len, target_len), binder_len, binder_len, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.add_patch(rect2)
    plt.savefig(f'{folder_name}/fold_cond_cmap.png')
    
    os.system(f'mkdir {folder_name}')
    np.save(f'{folder_name}/fold_cond_cmap.npy',fc_cmap)
    
    fc_cmap[fc_cmap>0] = 1
    np.save(f'{folder_name}/fold_cond_cmap_mask.npy',fc_cmap)

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

    # Create folders to save outputs
    os.system(f'mkdir {folder_name}/traj/')
    os.system(f'mkdir {folder_name}/mpnn/')
    os.system(f'mkdir {folder_name}/designs/')

    # Generate N number of trajectories
    if sample == False:
        
        for i in range(num_designs):
            i+=1
            clear_mem() # clearing memory at each step helps to avoid RunTimeError
            name = f'traj_{i}'
            rm_aa = 'C'
            
            af_model = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                                                 use_templates=True,)

            # load the fold-conditioned cmaps inside of af_model
            af_model.opt['cond_cmap'] = np.load(f'{folder_name}/fold_cond_cmap.npy')
            af_model.opt['cond_cmap_mask'] = np.load(f'{folder_name}/fold_cond_cmap_mask.npy')
            
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
        
            mpnn_model = mk_mpnn_model(model_name, backbone_noise=mpnn_backbone_noise,weights=mpnn_version)
            mpnn_model.prep_inputs(pdb_filename=f"{folder_name}/traj/{name}.pdb", chain='A,B', 
                                   fix_pos=sol_design_pos, rm_aa = "C", inverse=True)
        
            samples = mpnn_model.sample_parallel(temperature=mpnn_sampling_temp, batch=mpnn_samples)
            
            # save sequences in pickle file if --mpnn_save enabled
            if mpnn_save:
                with open(f'{folder_name}/mpnn/mpnn_{name}.pickle', 'wb') as handle:
                    pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)   
                
            #Predict Samples with AF2_ptm
            print('Predicting sequences with AF2_ptm...')    
            for num, seq in enumerate(samples['seq']):
                af_model = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                                                       use_templates=True,)
                af_model.opt['cond_cmap'] = np.load(f'{folder_name}/fold_cond_cmap.npy')
                af_model.opt['cond_cmap_mask'] = np.load(f'{folder_name}/fold_cond_cmap_mask.npy')
                af_model.prep_inputs(pdb_filename=pdb_target_path,
                                               chain=chain_id, binder_len = binder_len,
                                               hotspot=target_hotspots,
                                               rm_aa=rm_aa, #fix_pos=fixed_positions,
                                               )
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
    
    else:
        passed = 0
        success_target = 100
        i=0
        while passed <= success_target:
            clear_mem()
            i+=1
            name = f'traj_{i}' #@param {type:"string"}
            rm_aa = 'C' #@param {type:"string"}
            
            af_model = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                                                 use_templates=True,)
            
            af_model.opt['cond_cmap'] = np.load(f'{folder_name}/fold_cond_cmap.npy')
            af_model.opt['cond_cmap_mask'] = np.load(f'{folder_name}/fold_cond_cmap_mask.npy')
            
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
                    af_model = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                                                           use_templates=True,)
                    af_model.opt['cond_cmap'] = np.load(f'{folder_name}/fold_cond_cmap.npy')
                    af_model.opt['cond_cmap_mask'] = np.load(f'{folder_name}/fold_cond_cmap_mask.npy')
            
                    af_model.prep_inputs(pdb_filename=pdb_target_path,
                                                   chain=chain_id, binder_len = binder_len,
                                                   hotspot=target_hotspots,
                                                   rm_aa=rm_aa, #fix_pos=fixed_positions,
                                                   )
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
    
    df = pd.DataFrame({'name':names,
                               'sequence':sequences,
                               'plddt':plddts,
                               'ipae':ipaes,
                               'iptm':iptms,
                               'cmap_loss':cmap_loss})
    
    df.to_csv(f"{folder_name}/results.csv")

if __name__ == '__main__':
    main()