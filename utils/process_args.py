import argparse

def _process_args():
    r"""
    Function creates a namespace to read terminal-based arguments for running the experiment

    Args
        - None 

    Return:
        - args : argparse.Namespace

    """

    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    #---> study related
    parser.add_argument('--study', type=str, default='tcga_lihc', help='study name')
    parser.add_argument('--task', type=str, default='survival', choices=['survival'])
    parser.add_argument('--n_classes', type=int, default=4, help='number of classes (4 bins for survival)')
    parser.add_argument('--results_dir', default='F:/TCGA-LIHC-SLIDES/results', help='results directory (default: ./results)')
    parser.add_argument("--type_of_path", type=str, default="combine", choices=["xena", "hallmarks", "combine"])
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')

    #----> data related
    parser.add_argument('--data_root_dir', type=str, default='./pt_files', help='data directory')
    parser.add_argument('--label_file', type=str, default='./SurvPGC-main/datasets_csv/metadata/tcga_lihc.csv', help='Path to csv with labels')
    parser.add_argument('--omics_dir', type=str, default='./SurvPGC-main/datasets_csv/raw_rna_data/combine/lihc', help='Path to dir with omics csv for all modalities')
    parser.add_argument('--clinic_dir', type=str, default='./TCGA-LIHC-SLIDES/features_clinical_basic_1_template', help='Path to dir with clinical embedding')
    parser.add_argument('--gene_dir', type=str, default='./TCGA-LIHC-SLIDES/features_gene', help='Path to dir with gene foundation model embedding')

    parser.add_argument('--num_patches', type=int, default=4096, help='number of patches')
    parser.add_argument('--label_col', type=str, default="survival_months", help='type of survival (OS, DSS, PFI, PFS)')
    parser.add_argument("--wsi_projection_dim", type=int, default=256)
    parser.add_argument("--encoding_layer_1_dim", type=int, default=8)
    parser.add_argument("--encoding_layer_2_dim", type=int, default=16)
    parser.add_argument("--encoder_dropout", type=float, default=0.25)

    #----> split related 
    parser.add_argument('--k', type=int, default=5, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--split_dir', type=str, default='D:/PycharmProjects/SurvPath-main/splits/5foldcv/tcga_lihc2', help='manually specify the set of splits to use, '
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--which_splits', type=str, default="5foldcv", help='where are splits')
        
    #----> training related 
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--opt', type=str, default="radam", help="Optimizer")
    parser.add_argument('--reg_type', type=str, default="L2", help="regularization type [None, L1, L2]")
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--bag_loss', type=str, choices=['ce_surv', "nll_surv", "nll_rank_surv", "rank_surv", "cox_surv", "nll_diff_surv"], default='nll_surv',
                        help='survival loss function (default: ce)')
    parser.add_argument('--alpha_surv', type=float, default=0.5, help='weight given to uncensored patients')
    parser.add_argument('--beta_surv', type=float, default=0.3, help='weight given to diff_loss')
    parser.add_argument('--reg', type=float, default=0.001, help='weight decay / L2 (default: 0.001)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=1)

    #---> model related
    parser.add_argument('--fusion', type=str, default=None, choices=['concat', 'bilinear'])
    parser.add_argument('--modality', type=str, default="survpgc_f",
                        choices=['survpath', 'survpgc_f', 'omics', 'clinic_mlp', 'snn', 'clinic_snn', 'mlp_wsi', 'transmil_wsi', "abmil_wsi", 'survpc_f', 'survpc', 'coattn', 'porpoise', 'survpath_gc', 'mlppc_concat'])
    parser.add_argument('--return_attn', type=str, default=False, help="Used for heatmap drawing")
    parser.add_argument('--encoding_dim', type=int, default=1024, help='WSI encoding dim, default: 1024')
    parser.add_argument('--use_nystrom', action='store_true', default=False, help='Use Nystrom attentin in SurvPath.')

    args = parser.parse_args()

    if not (args.task == "survival"):
        print("Task and folder does not match")
        exit()

    return args