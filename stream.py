import numpy as np
import pandas as pd
import torch
from torch.utils import data
import json
import csv
from sklearn.preprocessing import OneHotEncoder
from time import time
from subword_nmt.apply_bpe import BPE
import codecs

vocab_path = './ESPF/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

vocab_path = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')

idx2word_d = sub_csv['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))


#def drug_sim_encoder2(x, path):
#    df = pd.DataFrame(pd.read_csv(path + 'drug_sim.csv'))
#    drug_sim = (df.loc[df["Unnamed: 0"] == x]).drop("Unnamed: 0", axis=1)
#    return drug_sim.values

#a = time()
#d = drug_sim_encoder2('DB00425', './dataset/BIOSNAP/full_data/test/')
#b = time()
#print(d)
#print(b-a)

#a = time()
#d = drug_sim_encoder('DB00425', './dataset/BIOSNAP/full_data/test/')
#b = time()
#print(d)
#print(b-a)


def drug_sim_encoder(x, path):
    drug_sim = []
    drug_sim_path = path + 'drug_sim.csv'
    with open(drug_sim_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            lens = len(line)
            if line[0] == str(x):
                i = 1
                while i < lens:
                    drug_sim.append(float(line[i]))
                    i = i + 1
    drug_sim = np.array(drug_sim)
    return drug_sim

def protein_sim_encoder(x, path):
    protein_sim = []
    protein_sim_path = path + 'target_sim.csv'
    with open(protein_sim_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            lens = len(line)
            if line[0] == x:
                i = 1
                while i < lens:
                    protein_sim.append(float(line[i]))
                    i = i + 1
    protein_sim = np.array(protein_sim)
    return protein_sim


max_d = 205
max_p = 545
def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask)


def drug2emb_encoder(x):
    max_d = 50
    # max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


class BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti, sim_path):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        self.sim_path = sim_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        # d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['SMILES']
        p = self.df.iloc[index]['Target Sequence']
        drug_id = self.df.iloc[index]['DrugBank ID']
        #protein_id = self.df.iloc[index]['Target ID']
        # d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)
        p_v, input_mask_p = protein2emb_encoder(p)

        drug_sim = drug_sim_encoder(drug_id, self.sim_path)
        protein_sim = protein_sim_encoder(p, self.sim_path)


        #print(drug_sim.shape)
        #print(protein_sim.shape)
        #print(d_v.shape)
        # print(input_mask_d.shape)
        #print(p_v.shape)
        # print(input_mask_p.shape)
        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, y, drug_sim, protein_sim