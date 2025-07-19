#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val_test
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment

from utils.process_args import _process_args
from warnings import simplefilter

simplefilter(action="ignore",category=FutureWarning)

def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    # folds = [4]
    
    #----> storing the val and test cindex for 5 fold cv
    all_test_cindex = []
    all_test_cindex_ipcw = []
    all_test_BS = []
    all_test_IBS = []
    all_test_iauc = []
    all_test_iauc_list = []
    all_test_loss = []

    for i in folds:
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train/val/test datasets for fold {}".format(i))

        results, (test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_iauc_list, total_loss), attn_matrix = _train_val_test(datasets, i, args)

        all_test_cindex.append(test_cindex)
        all_test_cindex_ipcw.append(test_cindex_ipcw)
        all_test_BS.append(test_BS)
        all_test_IBS.append(test_IBS)
        all_test_iauc.append(test_iauc)
        all_test_iauc_list.append(test_iauc_list)
        all_test_loss.append(total_loss)
    
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("Saving results...")
        _save_pkl(filename, results)
    
    final_df = pd.DataFrame({
        'test_cindex': all_test_cindex,
        'test_cindex_ipcw': all_test_cindex_ipcw,
        'test_IBS': all_test_IBS,
        'test_iauc': all_test_iauc,
        'test_iauc_list': all_test_iauc_list,
        "test_loss": all_test_loss,
        'test_BS': all_test_BS,
    })

    if len(folds) != args.k:
        save_name = 'test_result_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'test_result.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    start = timer()

    #----> read the args
    args = _process_args()
    
    #----> Prep
    args = _prepare_for_experiment(args)
    
    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed, 
        print_info=True, 
        n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat = True if "coattn" in args.modality else False,
        is_survpath = True if args.modality == "survpath" else False,
        is_survpgc = True if args.modality == "survpgc" else False,
        is_survpgc_f = True if args.modality == "survpgc_f" else False,
        is_survpc = True if args.modality == "survpc" else False,
        is_survpc_f = True if args.modality == "survpc_f" else False,
        type_of_pathway=args.type_of_path)


    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))