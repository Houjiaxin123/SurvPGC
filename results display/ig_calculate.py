#----> general imports
import pandas as pd
import numpy as np
import os
import argparse

import torch
from torch.utils.data import Dataset
from captum.attr import IntegratedGradients


parser = argparse.ArgumentParser(description='Configurations for cross-attention visualizing')

#---> study related
parser.add_argument('--n_bins', type=int, default=4, help='number of classes (4 bins for survival)')
parser.add_argument('--modality', type=str, default='survpgc_f', choices=['survpath', 'survpgc','survpgc_f'])
parser.add_argument('--label_col', type=str, default='survival_months', help='data directory')
parser.add_argument('--censorship_var', type=str, default='censorship', help='data directory')
parser.add_argument('--return_attn', type=str, default=False, help="Used for heatmap drawing")
parser.add_argument('--results_dir', default='F:/TCGA-COADREAD-SLIDES/results_revise', help='results directory (default: ./results)')
parser.add_argument('--bag_loss', default='nll_diff_surv')

#--->data load path
parser.add_argument("--model_path", type=str, default='F:/TCGA-COADREAD-SLIDES/results_revise/max_epoch/tcga_coadread__nll_surv_a0.0_lr5e-04_l2Weight_0.001_5foldcv_b1_survival_months_dim1_1024_patches_4096_wsiDim_256_epochs_20_fusion_None_modality_survpgc_f_pathT_combine/s_2_checkpoint.pt')
parser.add_argument('--sample_list', type=str, default='D:/PycharmProjects/SurvPath-main/splits/5foldcv/tcga_coadread2/splits_2.csv')
parser.add_argument('--clinic_path', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/clinical_data/tcga_coadread_clinical.csv')

parser.add_argument('--slide_dir', type=str, default='F:/TCGA-COADREAD-SLIDES/DX') #slides
parser.add_argument('--patches_dir', type=str, default='F:/TCGA-COADREAD-SLIDES/seg_DX/patches') #patch file
parser.add_argument('--data_root_dir', type=str, default='F:/TCGA-COADREAD-SLIDES/features_DX_UNI/pt_files', help='data directory') #patch festure file

parser.add_argument('--omics_dir', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/raw_rna_data/combine/coadread', help='Path to dir with omics csv for all modalities') #pathways
parser.add_argument('--gene_dir', type=str, default='F:/TCGA-COADREAD-SLIDES/features_gene') #gene foundation model feature
parser.add_argument('--clinic_dir', type=str, default='F:/TCGA-COADREAD-SLIDES/features_clinical_basic_1_template', help='Path to dir with clinical embedding') #clinic prompt feature
parser.add_argument('--label_file', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/metadata/tcga_coadread.csv', help='Path to csv with labels')


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


def _ig_test(datasets, args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if args.modality == 'survpgc_f':
        with torch.no_grad():
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic = _unpack_data(args.modality, device, datasets)
            input_args = {"x_path": data_WSI.unsqueeze(0).to(device)}
            input_args["x_omic"] = data_omics.unsqueeze(0).to(device)
            input_args["x_clinic"] = data_clinic.unsqueeze(0).to(device)
            input_args["return_attn"] = args.return_attn
            input_args["bag_loss"] = 'nll_diff_surv'

    tensor_clinic, tensor_gene, h = model(**input_args)

    return tensor_clinic, tensor_gene, h


def _calculate_risk(h):
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()


class PGCDataset(Dataset):
    def __init__(self,
                 modality,
                 patient_dict, #patient list
                 metadata, #clinical tabular
                 data_dir,
                 clinic_dir,
                 gene_dir,
                 num_classes
                 ):

        super(PGCDataset, self).__init__()

        self.modality = args.modality
        self.patient_dict = patient_dict
        self.metadata = metadata
        self.data_dir = data_dir
        self.clinic_dir = clinic_dir
        self.gene_dir = gene_dir
        self.num_classes = num_classes

    def __getitem__(self, idx):
        if args.modality == 'survpgc_f':
            label, event_time, c, slide_ids, clinin_data, case_id = self.get_data_to_return(idx)
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            clinic_features = self._load_clinic_embs_from_prompt(self.clinic_dir, slide_ids)
            gene_features = self._load_gene_embs_from_path(self.gene_dir, slide_ids)

            return (patch_features, gene_features, clinic_features, label, event_time, c, clinin_data, mask)

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
            clinic_bag = torch.load(feature_path)[1:]
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

        return gene_features

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



if __name__ == "__main__":
    # ----> read the args
    args = parser.parse_args()

    args.clinical_data = pd.read_csv(args.clinic_path, index_col=0)

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


    datasets = PGCDataset(modality=args.modality,
                 patient_dict=patient_dict, #patient list
                 metadata=label_data, #clinical tabular
                 data_dir=args.data_root_dir,
                 clinic_dir=args.clinic_dir,
                 gene_dir=args.gene_dir,
                 num_classes=8)

    model = torch.load(args.model_path, weights_only=False)
    model = model.eval()

    children = list(model.named_children())
    i = 0
    first_part = torch.nn.Sequential()
    second_part = torch.nn.Sequential()
    for name, module in children:
        i += 1
        if i <= 8:
            first_part.add_module(name, module)
        else:
            second_part.add_module(name, module)

    df = pd.DataFrame(columns=["patient_id", "risk", "survival", "label", "attr_c", "attr_g", "attr_c_abs", "attr_g_abs"])
    all_split = pd.read_csv(args.sample_list)
    sample_list = list((all_split["test"]).dropna(axis=0, how='any'))
    # sample_list = ['TCGA-2Y-A9GT']

    for patient in sample_list:
        if patient in datasets.patient_dict:
            index = list(datasets.patient_dict).index(patient)
            file_name = str(datasets.patient_dict[patient][0][:-4])
            data = datasets[index]

            tensor_c, tensor_g, h = _ig_test(data, args, model)
            clinic = torch.chunk(tensor_c, 2, dim=1)
            tensor_c1 = clinic[0]
            tensor_c2 = clinic[1]
            gene = torch.chunk(tensor_g, 2, dim=1)
            tensor_g1 = gene[0]
            tensor_g2 = gene[1]

            embedding1 = torch.cat([tensor_c1, tensor_c2, tensor_g1, tensor_g2], dim=1)
            embedding2 = torch.cat([tensor_c1, tensor_c2, tensor_g1, tensor_g2], dim=0)

            risk, survival = _calculate_risk(h)
            integrated_grandients = IntegratedGradients(second_part)

            target = int(data[3])

            attributions_ig = integrated_grandients.attribute(embedding1, target=target)
            attr = torch.chunk(attributions_ig, 2, dim=1)
            attr_c = attr[0]
            attr_g = attr[1]
            cum_attr_c = attr_c.sum().detach().cpu().numpy()
            cum_attr_g = attr_g.sum().detach().cpu().numpy()
            cum_attr_c_ab = attr_c.abs().sum().detach().cpu().numpy()
            cum_attr_g_ab = attr_g.abs().sum().detach().cpu().numpy()

            df = df._append({"patient_id":patient,
                             "risk":risk,
                             "survival":survival,
                             "label":target,
                             "attr_c":cum_attr_c,
                             "attr_g":cum_attr_g,
                             "attr_c_abs":cum_attr_c_ab,
                             "attr_g_abs":cum_attr_g_ab},
                            ignore_index=True)
            print("Finish:", patient)

    df.to_csv("F:/TCGA-COADREAD-SLIDES/results_revise/ig_attribution.csv", index=False)
    print("Finish")










