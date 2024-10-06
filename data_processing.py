import os
import glob
import _pickle as cPickle
import copy
import argparse
import csv
import pandas as pd
from decimal import *

from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
import Levenshtein

# Transform the drug_tar data into graph
def Trans_drug_tar(filename, output, len):
    row = []
    col = []
    label = []
    data = []

    with open(filename) as f:
        reader = csv.reader(f)
        for line in reader:
            row.append(line[0])
            col.append(line[1])
            label.append(line[2])

    data = [[0 for i in range(len)] for i in range(len)]

    i = 0
    flag = 0
    while i < len:
        j = 0
        while j < len:
            if i==j:
                data[i][j] = label[flag]
                flag = flag + 1
            j = j + 1
        i = i + 1

    with open(output, "w", encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        for line in data:
            writer.writerow(line)


# Calculate drug drug similarity
def calculate_drug_similarity(input_file, output_file):
    input_drug_info = {}
    base_drug_info = {}
    with open(input_file, 'r')as fp:
        reader = csv.reader(fp)
        for line in reader:
            drug1 = line[0]
            smiles1 = line[1]
            drug2 = line[2]
            smiles2 = line[3]
            if drug1 not in input_drug_info:
                input_drug_info[drug1] = smiles1
            if drug2 not in base_drug_info:
                base_drug_info[drug2] = smiles2

    drug_similarity_info = {}
    i = 0
    for input_drug_id in input_drug_info:
        each_smiles = input_drug_info[input_drug_id]
        drug1_mol = Chem.MolFromSmiles(each_smiles)
        drug1_mol = AllChem.AddHs(drug1_mol)

        drug_similarity_info[input_drug_id] = {}

        for input_drug_id2 in base_drug_info:
            each_smiles = base_drug_info[input_drug_id2]
            if each_smiles:
                drug2_mol = Chem.MolFromSmiles(each_smiles)
                drug2_mol = AllChem.AddHs(drug2_mol)
                fps = AllChem.GetMorganFingerprint(drug1_mol, 2)
                fps2 = AllChem.GetMorganFingerprint(drug2_mol, 2)
                score = DataStructs.DiceSimilarity(fps, fps2)
                drug_similarity_info[input_drug_id][input_drug_id2] = score
                #print("Drug：processing " + str(input_drug_id) + "and " + str(input_drug_id2))

    df = pd.DataFrame.from_dict(drug_similarity_info)
    df.T.to_csv(output_file)


# Calculate protein protein similarity
def calculate_protein_similarity(input_file, output_file):
    input_protein_info = {}
    base_protein_info = {}
    with open(input_file, 'r')as fp:
        reader = csv.reader(fp)
        for line in reader:
            label1 = line[0]
            protein1 = line[0]
            label2 = line[1]
            protein2 = line[1]
            if label1 not in input_protein_info:
                input_protein_info[label1] = protein1
            if label2 not in base_protein_info:
                base_protein_info[label2] = protein2

    protein_similarity_info = {}
    i = 0
    for input_protein_id in input_protein_info:
        protein1 = input_protein_info[input_protein_id]

        protein_similarity_info[input_protein_id] = {}

        for input_protein_id2 in base_protein_info:
            protein2 = base_protein_info[input_protein_id2]
            ratio = Levenshtein.ratio(protein1, protein2)
            protein_similarity_info[input_protein_id][input_protein_id2] = ratio
            #print("Protein：processing " + str(input_protein_id) + "and " + str(input_protein_id2))
            i = i + 1
            print(i)
    df = pd.DataFrame.from_dict(protein_similarity_info)
    df.T.to_csv(output_file)

#Test
print("Test data Processing...")
drug_input_file = './pre/input/data_input/drug_input.csv'
drug_similarity_profile = './pre/input/drug_sim.csv'

protein_input_file = './pre/input/data_input/protein_input.csv'
protein_similarity_profile = './pre/input/target_sim.csv'

calculate_drug_similarity(drug_input_file, drug_similarity_profile)
calculate_protein_similarity(protein_input_file, protein_similarity_profile)
print("Test data done!")
#
# #train
# print("Train data Processing...")
# drug_input_file = '../moltrans/dataset/BIOSNAP/full_data/train/input_sim/drug_input.csv'
# drug_similarity_profile = '../moltrans/dataset/BIOSNAP/full_data/train/drug_sim.csv'
#
# protein_input_file = '../moltrans/dataset/BIOSNAP/full_data/train/input_sim/target_input.csv'
# protein_similarity_profile = '../moltrans/dataset/BIOSNAP/full_data/train/target_sim.csv'
#
# calculate_drug_similarity(drug_input_file, drug_similarity_profile)
# calculate_protein_similarity(protein_input_file, protein_similarity_profile)
# print("Train data done!")
#
# #val
# print("Val data Processing...")
# drug_input_file = '../moltrans/dataset/BIOSNAP/full_data/val/input_sim/drug_input.csv'
# drug_similarity_profile = '../moltrans/dataset/BIOSNAP/full_data/val/drug_sim.csv'
#
# protein_input_file = '../moltrans/dataset/BIOSNAP/full_data/val/input_sim/target_input.csv'
# protein_similarity_profile = '../moltrans/dataset/BIOSNAP/full_data/val/target_sim.csv'
#
# calculate_drug_similarity(drug_input_file, drug_similarity_profile)
# calculate_protein_similarity(protein_input_file, protein_similarity_profile)
# print("Val data done!")

#filename = '../data/drug_tar_input.csv'
#output = '../data/drug_tar.csv'
#len = 548
#Trans_drug_tar(filename, output, len)  #len是数据长度