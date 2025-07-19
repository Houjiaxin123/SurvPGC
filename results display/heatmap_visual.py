#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
import argparse

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.general_utils import _series_intersection
from sklearn.preprocessing import MinMaxScaler
from wsi_core.WholeSlideImage import WholeSlideImage
import h5py
from wsi_core.wsi_utils import sample_rois


parser = argparse.ArgumentParser(description='Configurations for cross-attention visualizing')

#---> study related
parser.add_argument('--n_bins', type=int, default=4, help='number of classes (4 bins for survival)')
parser.add_argument('--modality', type=str, default='survpgc_f', choices=['survpath', 'survpgc', 'survpgc_f'])
parser.add_argument('--bag_loss', type=str, default='nll_surv', choices=['nll_surv', 'nll_diff_surv'])
parser.add_argument('--label_col', type=str, default='survival_months', help='data directory')
parser.add_argument('--censorship_var', type=str, default='censorship', help='data directory')
parser.add_argument('--return_attn', type=str, default=True, help="Used for heatmap drawing")
parser.add_argument('--results_dir', default='F:/TCGA-LIHC-SLIDES/results_revise', help='results directory (default: ./results)')
parser.add_argument('--heatmap_result_dir', type=str, default='F:/TCGA-LIHC-SLIDES/results_revise/heatmap/image')
parser.add_argument('--top_patch_dir', default='F:/TCGA-LIHC-SLIDES/results_revise/heatmap/patch_top')

#--->data load path
parser.add_argument("--model_path", type=str, default='F:/TCGA-LIHC-SLIDES/results_revise/max_epoch/tcga_lihc__nll_surv_a0.0_lr5e-04_l2Weight_0.001_5foldcv_b1_survival_months_dim1_1024_patches_4096_wsiDim_256_epochs_20_fusion_None_modality_survpgc_f_pathT_combine/s_2_checkpoint.pt')
parser.add_argument('--sample_list', type=str, default='F:/TCGA-LIHC-SLIDES/results_revise/heatmap/heatmap_list.csv')
parser.add_argument('--clinic_path', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/clinical_data/tcga_lihc_clinical.csv')

parser.add_argument('--slide_dir', type=str, default='F:/TCGA-LIHC-SLIDES/DX') #slides
parser.add_argument('--patches_dir', type=str, default='F:/TCGA-LIHC-SLIDES/seg_DX/patches') #patch file
parser.add_argument('--data_root_dir', type=str, default='F:/TCGA-LIHC-SLIDES/features_DX_UNI/pt_files', help='data directory') #patch festure file

parser.add_argument('--omics_dir', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/raw_rna_data/combine/lihc', help='Path to dir with omics csv for all modalities') #pathways
parser.add_argument('--gene_dir', type=str, default='F:/TCGA-LIHC-SLIDES/features_gene') #gene foundation model feature
parser.add_argument('--clinic_dir', type=str, default='F:/TCGA-LIHC-SLIDES/features_clinical_basic_1_template', help='Path to dir with clinical embedding') #clinic prompt feature
parser.add_argument('--label_file', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/metadata/tcga_lihc.csv', help='Path to csv with labels')

# heatmap related
parser.add_argument('--vis_level', type=int, default=2)
parser.add_argument('--alpha', type=float, default=0.3)
parser.add_argument('--blank_canvas', default=False)
parser.add_argument('--save_orig', default=True)
parser.add_argument('--save_ext', default='jpg')
parser.add_argument('--use_ref_scores', default=False)
parser.add_argument('--blur', default=False)
parser.add_argument('--use_center_shift', default=True)
parser.add_argument('--use_roi', default=False)
parser.add_argument('--calc_heatmap', default=True)
parser.add_argument('--binarize', default=False)
parser.add_argument('--binary_thresh', default=-1)
parser.add_argument('--custom_downsample', default=1)
parser.add_argument('--cmap', default='jet')
parser.add_argument('--segment', default=False)

# attention samples
parser.add_argument('--sample', default=True)
parser.add_argument('--sample_name', type=str, default='topk_high_attention')
parser.add_argument('--sample_seed', type=int, default=1)
parser.add_argument('--sample_k', type=int, default=50)
parser.add_argument('--sample_mode', default='topk')


ALL_MODALITIES = ['rna_clean.csv']


def _discretize_survival_months(eps, label_data):
    # cut the data into self.n_bins (4= quantiles)
    disc_labels, q_bins = pd.qcut(label_data[args.label_col], q=args.n_bins, retbins=True, labels=False)
    q_bins[-1] = label_data[args.label_col].max() + eps
    q_bins[0] = label_data[args.label_col].min() - eps

    disc_labels, q_bins = pd.cut(label_data[args.label_col], bins=q_bins, retbins=True, labels=False, right=False,
                                 include_lowest=True)
    label_data.insert(2, 'label', disc_labels.values.astype(int))
    args.bins = q_bins

    return label_data


def _apply_scaler(data, scaler):
    # find out which values are missing
    zero_mask = data == 0
    # transform data
    transformed = scaler.transform(data)
    data = transformed
    # rna -> put back in the zeros
    data[zero_mask] = 0.

    return data


def _unpack_data(modality, device, data):
    if modality == 'survpgc_f':
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        data_clinic = data[2].to(device)

        if data[7][0, 0] == 1:
            mask = None
        else:
            mask = data[7].to(device)

        y_disc, event_time, censor, clinical_data_list = data[3], data[4], data[5], data[6]

        y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

        return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic

    else:
        raise NotImplementedError('Model Type [%s] not implemented.' %modality)


def _heatmap_test(datasets, args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    args.return_attn = True

    if args.modality == 'survpgc_f':
        with torch.no_grad():
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic = _unpack_data(args.modality, device, datasets)
            input_args = {"x_path": data_WSI.unsqueeze(0).to(device)}
            input_args["x_omic"] = data_omics.unsqueeze(0).to(device)
            input_args["x_clinic"] = data_clinic.unsqueeze(0).to(device)
            input_args["return_attn"] = args.return_attn
            input_args["bag_loss"] = args.bag_loss

    h, App, Agp, Apg, Acc, Acp, Apc = model(**input_args)

    return h, App, Agp, Apg, Acc, Acp, Apc


def _calculate_risk(h):
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def attn_normlization(attn):
    attn = attn.detach().cpu().numpy()
    mean0 = np.mean(attn, axis=1, keepdims=True)
    std0 = np.std(attn, axis=1, keepdims=True)

    attn_norm = (attn - mean0) / std0

    return attn_norm


class PGCDataset(Dataset):
    def __init__(self,
                 modality,
                 patient_dict, #patient list
                 metadata, #clinical tabular
                 omics_data_dict, #
                 data_dir,
                 clinic_dir,
                 gene_dir,
                 num_classes,
                 omic_names,
                 num_pathways):

        super(PGCDataset, self).__init__()

        self.modality = args.modality
        self.patient_dict = patient_dict
        self.metadata = metadata
        self.omics_data_dict = omics_data_dict
        self.data_dir = data_dir
        self.clinic_dir = clinic_dir
        self.gene_dir = gene_dir
        self.num_classes = num_classes
        self.omic_names = omic_names
        self.num_pathways = 331

    def __getitem__(self, idx):
        if args.modality == 'survpgc_f':
            label, event_time, c, slide_ids, clinin_data, case_id = self.get_data_to_return(idx)
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            clinic_features = self._load_clinic_embs_from_prompt(self.clinic_dir, slide_ids)
            gene_features = self._load_gene_embs_from_path(self.gene_dir, slide_ids)

            return (patch_features, gene_features, clinic_features, label, event_time, c, clinin_data, mask)

        else:
            raise NotImplementedError('Model Type [%s] not implemented.' % args.modality)

    def get_data_to_return(self, idx):
        case_id = self.metadata['case_id'][idx]
        label = torch.Tensor([self.metadata['disc_label'][idx]])  # disc
        event_time = torch.Tensor([self.metadata['survival_months'][idx]])
        c = torch.Tensor([self.metadata['censorship'][idx]])
        slide_ids = self.patient_dict[case_id]
        clinical_data = self.get_clinical_data(case_id)

        return label, event_time, c, slide_ids, clinical_data, case_id

    def _load_wsi_embs_from_path(self, data_dir, slide_ids):
        patch_features = []
        # load all slide_ids corresponding for the patient
        for slide_id in slide_ids:
            wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi_path)
            patch_features.append(wsi_bag)
        patch_features = torch.cat(patch_features, dim=0)
        mask = torch.ones([1, 1])

        return patch_features, mask

    def _load_clinic_embs_from_prompt(self, clinic_dir, slide_ids):
        clinic_features = []
        for sample in slide_ids:
            feature_path = os.path.join(clinic_dir, '{}.pt'.format(sample[0:12]))
            clinic_bag = torch.load(feature_path)
            clinic_features.append(clinic_bag)
        clinic_features = torch.cat(clinic_features, dim=0)

        return clinic_features

    def _load_gene_embs_from_path(self, gene_dir, slide_ids):
        gene_features = []
        for sample in slide_ids:
            feature_path = os.path.join(gene_dir, '{}.pt'.format(sample[0:12]))
            gene_bag = torch.load(feature_path)
            gene_features.append(gene_bag)
        gene_features = torch.cat(gene_features, dim=0)

        return  gene_features

    def get_clinical_data(self, case_id):
        try:
            stage = args.clinical_data.loc[case_id, "stage"]
        except:
            stage = "N/A"
        try:
            grade = args.clinical_data.loc[case_id, "grade"]
        except:
            grade = "N/A"
        try:
            subtype = args.clinical_data.loc[case_id, "subtype"]
        except:
            subtype = "N/A"

        clinical_data = (stage, grade, subtype)
        return clinical_data

    def __len__(self):
        return len(self.metadata)


def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level=-1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)

    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)

    return heatmap


if __name__ == "__main__":
    # ----> read the args
    args = parser.parse_args()
    # ---> perform the experiment
    # rna 4999 for all patients
    all_modalities = {}
    for modality in ALL_MODALITIES:
        all_modalities[modality.split('_')[0]] = pd.read_csv(
            os.path.join(args.omics_dir, modality),
            engine='python',
            index_col=0
        )

    heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': args.vis_level, 'blur': args.blur, 'segment': args.segment,
                        'custom_downsample': args.custom_downsample}

    args.clinical_data = pd.read_csv(args.clinic_path, index_col=0)

    # the number of genes in each pathway
    signatures = pd.read_csv("./datasets_csv/metadata/combine_signatures.csv")
    omic_names = []
    for col in signatures.columns:
        omic = signatures[col].dropna().unique()
        omic = sorted(_series_intersection(omic, all_modalities["rna"].columns))
        omic_names.append(omic)
    omic_sizes = [len(omic) for omic in omic_names]

    # patient label
    label_data = pd.read_csv(args.label_file, low_memory=False)

    label_data_re = _discretize_survival_months(1e-6, label_data)
    label_dict = {}
    key_count = 0
    for i in range(len(args.bins) - 1):
        for c in [0, 1]:
            label_dict.update({(i, c): key_count})
            key_count += 1

    for i in label_data_re.index:
        key = label_data_re.loc[i, 'label']
        label_data_re.at[i, 'disc_label'] = key
        censorship = label_data_re.loc[i, args.censorship_var]
        key = (key, int(censorship))
        label_data_re.at[i, 'label'] = label_dict[key]

    args.num_classes = len(label_dict)
    args.label_dict = label_dict
    patients_df = label_data
    patient_data = {'case_id': patients_df["case_id"].values,
                    'label': patients_df['label'].values}  # only setting the final data to self

    # patient list
    patient_dict = {}
    temp_label_data = label_data.set_index('case_id')
    for patient in patients_df['case_id']:
        slide_ids = temp_label_data.loc[patient, 'slide_id']
        if isinstance(slide_ids, str):
            slide_ids = np.array(slide_ids).reshape(-1)
        else:
            slide_ids = slide_ids.values
        patient_dict.update({patient: slide_ids})

    #omic data and normalization
    omics_data = {}
    for key in all_modalities.keys():
        raw_data_df = all_modalities[key]
        raw_data_df["temp_index"] = raw_data_df.index
        raw_data_df.reset_index(inplace=True, drop=True)

        # normalize your df
        filtered_normed_df = None
        case_ids = raw_data_df["temp_index"]
        df_for_norm = raw_data_df.drop(labels="temp_index", axis=1)

        num_patients = df_for_norm.shape[0]
        num_feats = df_for_norm.shape[1]
        columns = {}
        for i in range(num_feats):
            columns[i] = df_for_norm.columns[i]

        flat_df = np.expand_dims(df_for_norm.values.flatten(), 1)
        scaler_for_data = MinMaxScaler(feature_range=(-1, 1)).fit(flat_df)
        normed_flat_df = _apply_scaler(data=flat_df, scaler=scaler_for_data)
        filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))
        filtered_normed_df["temp_index"] = case_ids
        filtered_normed_df.rename(columns=columns, inplace=True)

        omics_data[key] = filtered_normed_df

    datasets = PGCDataset(modality=args.modality,
                 patient_dict=patient_dict, #patient list
                 metadata=label_data, #clinical tabular
                 omics_data_dict=omics_data, #omic data
                 data_dir=args.data_root_dir,
                 clinic_dir=args.clinic_dir,
                 gene_dir=args.gene_dir,
                 num_classes=8,
                 omic_names=omic_names,
                 num_pathways=len(omic_names))

    model = torch.load(args.model_path, weights_only=False)

    all_samples = pd.read_csv(args.sample_list)
    sample_list = list(all_samples['patient_id'])

    for patient in sample_list:
        if patient in datasets.patient_dict:
            index = list(datasets.patient_dict).index(patient)
            file_name = str(datasets.patient_dict[patient][0][:-4])
            data = datasets[index]
            h, App, Agp, Apg, Acc, Acp, Apc = _heatmap_test(data, args, model)
            risk, survival = _calculate_risk(h)

            coord_file_path = os.path.join(args.patches_dir, str(file_name)+'.h5')
            coord_file = h5py.File(coord_file_path, "r")
            coords = coord_file['coords'][:]

            slide_path = os.path.join(args.slide_dir, str(file_name)+'.svs')
            wsi_object = WholeSlideImage(slide_path)

            Acp_n = attn_normlization(Acp)
            Agp_n = attn_normlization(Agp)

            if args.sample:
                sample_save_dir = os.path.join(args.top_patch_dir, patient)
                os.makedirs(sample_save_dir)

                scores_total_Acp = Acp_n.sum(axis=0)/Acp_n.shape[0]
                sample_results_Acp = sample_rois(scores_total_Acp, coords, k=args.sample_k, mode=args.sample_mode, seed=args.sample_seed)
                for idx, (s_coord, s_score) in enumerate(
                        zip(sample_results_Acp['sampled_coords'], sample_results_Acp['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), 0, (512, 512)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir, 'Acp_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, s_coord[0], s_coord[1], s_score)))

                scores_total_Agp = Agp_n.sum(axis=0) / Agp_n.shape[0]
                sample_results_Agp = sample_rois(scores_total_Agp, coords, k=args.sample_k, mode=args.sample_mode, seed=args.sample_seed)
                for idx, (s_coord, s_score) in enumerate(
                        zip(sample_results_Agp['sampled_coords'], sample_results_Agp['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), 0, (512, 512)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir, 'Agp_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, s_coord[0], s_coord[1], s_score)))

            heatmap_Acp = drawHeatmap(Acp_n, coords, slide_path, wsi_object=wsi_object,
                          cmap=args.cmap, alpha=args.alpha, **heatmap_vis_args,
                          binarize=args.binarize,
                          blank_canvas=args.blank_canvas,
                          thresh=args.binary_thresh, patch_size=512,
                          overlap=0)

            heatmap_Agp = drawHeatmap(Agp_n, coords, slide_path, wsi_object=wsi_object,
                          cmap=args.cmap, alpha=args.alpha, **heatmap_vis_args,
                          binarize=args.binarize,
                          blank_canvas=args.blank_canvas,
                          thresh=args.binary_thresh, patch_size=512,
                          overlap=0)

            heatmap_save_path = os.path.join(args.heatmap_result_dir, patient)
            os.makedirs(heatmap_save_path)
            if args.save_ext == 'jpg':
                heatmap_Acp.save(os.path.join(heatmap_save_path, str(patient)+'_Acp.jpg'), quality=100)
                heatmap_Agp.save(os.path.join(heatmap_save_path, str(patient)+'_Agp.jpg'), quality=100)





