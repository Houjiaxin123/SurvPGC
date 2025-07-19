#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
import argparse

import torch
from torch.utils.data import Dataset
from utils.general_utils import _series_intersection
from sklearn.preprocessing import MinMaxScaler


parser = argparse.ArgumentParser(description='Configurations for risk calculation of different models')

#---> study related
parser.add_argument('--n_bins', type=int, default=4, help='number of classes (4 bins for survival)')
parser.add_argument('--modality', type=str, default='transmil_wsi', choices=['survpath', 'survpgc_f', 'coattn', 'porpoise', 'transmil_wsi', 'omics', 'clinic_mlp'])
parser.add_argument('--label_col', type=str, default='survival_months', help='data directory')
parser.add_argument('--censorship_var', type=str, default='censorship', help='data directory')
parser.add_argument('--return_attn', type=str, default=False, help="Used for heatmap drawing")
parser.add_argument('--bag_loss', default='nll_surv')
parser.add_argument('--fusion', type=str, default=None, choices=['concat', 'bilinear'])

#--->data load path
parser.add_argument("--model_path", type=str, default='F:/TCGA-LIHC-SLIDES/results_revise/last_epoch/tcga_lihc__nll_surv_a0.5_lr1e-03_l2Weight_0.0001_5foldcv_b1_survival_months_dim1_1024_patches_4096_wsiDim_256_epochs_20_fusion_None_modality_transmil_wsi_pathT_combine/s_0_checkpoint.pt')
parser.add_argument('--sample_list', type=str, default='D:/PycharmProjects/SurvPath-main/splits/5foldcv/tcga_lihc/splits_0_val.csv')
parser.add_argument('--clinic_path', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/clinical_data/tcga_lihc_clinical.csv')

parser.add_argument('--slide_dir', type=str, default='F:/TCGA-LIHC-SLIDES/DX') #slides
parser.add_argument('--patches_dir', type=str, default='F:/TCGA-LIHC-SLIDES/seg_DX/patches') #patch file
parser.add_argument('--data_root_dir', type=str, default='F:/TCGA-LIHC-SLIDES/features_DX_UNI/pt_files', help='data directory') #patch festure file

parser.add_argument('--omics_dir', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/raw_rna_data/combine/lihc', help='Path to dir with omics csv for all modalities') #pathways
parser.add_argument('--gene_dir', type=str, default='F:/TCGA-LIHC-SLIDES/features_gene') #gene foundation model feature
parser.add_argument('--clinic_dir', type=str, default='F:/TCGA-LIHC-SLIDES/features_clinical_basic_1_template', help='Path to dir with clinical embedding') #clinic prompt feature
parser.add_argument('--label_file', type=str, default='D:/PycharmProjects/SurvPath-main/datasets_csv/metadata/tcga_lihc.csv', help='Path to csv with labels')

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
    if modality == 'omics_f':
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        data_clinic = None
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        data_clinic = None
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality == 'clinic_mlp':
        data_WSI = data[0]
        mask = None
        data_clinic = data[1].to(device)
        data_omics = None
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality == 'transmil_wsi':
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        data_clinic = None

        if data[6][0, 0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality == "coattn":
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]
        data_clinic = None

        y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality == "porpoise":
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        data_clinic = None
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality == 'survpc_f':
        data_WSI = data[0].to(device)
        data_clinic = data[1].to(device)
        data_omics = None
        if data[6][0, 0] == 1:
            mask = None
        else:
            mask = data[6].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality == 'survpath':
        data_WSI = data[0].to(device)
        data_omics = []
        for item in data[1]:
            data_omics.append(item.to(device))
        if data[6][0, 0] == 1:
            mask = None
        else:
            mask = data[6].to(device)
        data_clinic = None
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality == 'survpgc_f':
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        data_clinic = data[2].to(device)
        if data[7][0, 0] == 1:
            mask = None
        else:
            mask = data[7].to(device)
        y_disc, event_time, censor, clinical_data_list = data[3], data[4], data[5], data[6]
        y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    else:
        raise ValueError('Unsupported modality:', modality)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic



def _risk_test(data, args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    args.return_attn = False

    with torch.no_grad():
        data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, data_clinic = _unpack_data(args.modality, device, data)

        if args.modality == 'omics_f':
            input_args = {"x_path": data_WSI.to(device)}
            input_args["x_omic"] = data_omics.to(device)
            input_args["return_attn"] = args.return_attn
            h = model(**input_args)

        if args.modality in ['omics', 'transmil_wsi']:
            h = model(data_omics=data_omics,data_WSI=data_WSI,mask=mask)

        if args.modality == 'clinic_mlp':
            input_args = {"x_path": data_WSI.to(device)}
            input_args['x_clinic'] = data_clinic.to(device)
            input_args["return_attn"] = args.return_attn
            h = model(**input_args)

        if args.modality == 'survpc_f':
            input_args = {"x_path": data_WSI.to(device)}
            input_args["x_clinic"] = data_clinic.to(device)
            input_args["return_attn"] = args.return_attn
            h = model(**input_args)

        if args.modality == 'porpoise':
            input_args = {"x_path": data_WSI.to(device)}
            input_args["x_omic"] = data_omics.unsqueeze(0).to(device)
            h = model(**input_args)

        if args.modality == 'coattn':
            h = model(
                x_path=data_WSI,
                x_omic1=data_omics[0],
                x_omic2=data_omics[1],
                x_omic3=data_omics[2],
                x_omic4=data_omics[3],
                x_omic5=data_omics[4],
                x_omic6=data_omics[5]
            )

        if args.modality == 'survpath':
            input_args = {"x_path": data_WSI.to(device)}
            for i in range(len(data_omics)):
                input_args['x_omic%s' % str(i + 1)] = data_omics[i].type(torch.FloatTensor).to(device)
            input_args["return_attn"] = False
            h = model(**input_args)

        if args.modality == 'survpgc_f':
            input_args = {"x_path": data_WSI.unsqueeze(0).to(device)}
            input_args["x_omic"] = data_omics.unsqueeze(0).to(device)
            input_args["x_clinic"] = data_clinic.unsqueeze(0).to(device)
            input_args["return_attn"] = args.return_attn
            input_args["bag_loss"] = args.bag_loss
            h = model(**input_args)

    return h


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
                 omics_data_dict, #
                 data_dir,
                 clinic_dir,
                 gene_dir,
                 num_classes,
                 omic_names,
                 num_pathways,
                 is_mcat=False,
                 is_survpath=False,
                 is_survpgc=False
                 ):

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
        self.is_mcat = is_mcat
        self.is_survpath = is_survpath
        self.is_survpgc = is_survpgc
        self._setup_omics_data()

        if self.is_mcat:
            self._setup_mcat()
        elif self.is_survpath or self.is_survpgc:
            self._setup_survpath()
        else:
            self.omic_names = []
            self.omic_sizes = []

    def _setup_omics_data(self):
        self.all_modalities = {}
        for modality in ALL_MODALITIES:
            self.all_modalities[modality.split('_')[0]] = pd.read_csv(
                os.path.join(args.omics_dir, modality),
                engine='python',
                index_col=0
            )

    def _setup_mcat(self):
        self.signatures = pd.read_csv("./datasets_csv/metadata/signatures.csv")
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

    def _setup_survpath(self):
        self.signatures = pd.read_csv("./datasets_csv/metadata/combine_signatures.csv")
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

    def __getitem__(self, idx):
        label, event_time, c, slide_ids, clinical_data, case_id = self.get_data_to_return(idx)
        if args.modality in ['omics', 'snn']:
            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            return (torch.zeros((1,1)), omics_tensor, label, event_time, c, clinical_data)

        if args.modality == 'omics_f':
            gene_features = self._load_gene_embs_from_path(self.gene_dir, slide_ids)
            return (torch.zeros((1, 1)), gene_features, label, event_time, c, clinical_data)

        if args.modality == 'clinic_mlp':
            clinic_features = self._load_clinic_embs_from_prompt(self.clinic_dir, slide_ids)
            return (torch.zeros((1, 1)), clinic_features, label, event_time, c, clinical_data)

        if args.modality == 'transmil_wsi':
            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            # @HACK: returning case_id, remove later
            return (patch_features, omics_tensor, label, event_time, c, clinical_data, mask)

        if args.modality == 'porpoise':
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            return (patch_features, omics_tensor, label, event_time, c, clinical_data, mask)

        if args.modality == 'coattn':
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            omic1 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[0]].iloc[idx])
            omic2 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[1]].iloc[idx])
            omic3 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[2]].iloc[idx])
            omic4 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[3]].iloc[idx])
            omic5 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[4]].iloc[idx])
            omic6 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[5]].iloc[idx])
            return (patch_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data, mask)

        if args.modality == 'survpc_f':
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            clinic_features = self._load_clinic_embs_from_prompt(self.clinic_dir, slide_ids)
            return (patch_features, clinic_features, label, event_time, c, clinical_data, mask)

        if args.modality == 'survpath':
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            omic_list = []
            for i in range(self.num_pathways):
                omic_list.append(torch.tensor(self.omics_data_dict["rna"][self.omic_names[i]].iloc[idx]))
            return (patch_features, omic_list, label, event_time, c, clinical_data, mask)

        if args.modality == 'survpgc_f':
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            clinic_features = self._load_clinic_embs_from_prompt(self.clinic_dir, slide_ids)
            gene_features = self._load_gene_embs_from_path(self.gene_dir, slide_ids)
            return (patch_features, gene_features, clinic_features, label, event_time, c, clinical_data, mask)

        else:
            raise ValueError('Unsupported modality:', modality)

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
                 num_pathways=len(omic_names),
                is_mcat=False,
                is_survpath=False,
                is_survpgc=False,
                )

    model = torch.load(args.model_path, weights_only=False)
    model = model.eval()

    df = pd.DataFrame(columns=["patient_id", "survival_months", "censorship", "risk"])
    all_samples = pd.read_csv(args.sample_list)
    sample_list = list(all_samples['patient_id'])
    # sample_list = ['TCGA-XX-A89A']

    for patient in sample_list:
        if patient in datasets.patient_dict:
            index = list(datasets.patient_dict).index(patient)
            file_name = str(datasets.patient_dict[patient][0][:-4])
            data = datasets[index]

            h = _risk_test(data, args, model)
            # h = h.unsqueeze(0)

            risk, survival = _calculate_risk(h)
            survival_months = data[3].detach().numpy()
            censorship = data[4].detach().numpy()

            df = df._append({"patient_id": patient,
                             "survival_months": float(survival_months),
                             "censorship": int(censorship),
                             "risk": float(risk)},
                            ignore_index=True)
            print("Finish:", patient)

    df.to_csv("F:/TCGA-LIHC-SLIDES/results_revise/predict_risk_lihc/risk_split_0_transmil.csv", index=False)
    print("Finish")




