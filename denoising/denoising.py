from __future__ import print_function
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_array', default='./151673_cor/exp_array.csv')
parser.add_argument('--pos_adj', default='./151673_cor/pos_adj.csv')
parser.add_argument('--pos_dis', default='./151673_cor/pos_dis.csv')
parser.add_argument('--image_adj', default='./151673_cor/adj_151673_inception_v3_cos.csv')
parser.add_argument('--image_dis', default='./151673_cor/cor_151673_inception_v3_cos.csv')
parser.add_argument('--destination', default='./151673_cor/output_inception_v3_cos.csv')
args = parser.parse_args()

print(args.destination)

exp = pd.read_csv(args.exp_array, header = None, low_memory = False)
exp = np.array(exp)
exp_header = exp[0]
exp_header = exp_header[1:exp_header.shape[0]]
exp_index = exp[:,0]
exp_index = exp_index[1:exp_index.shape[0]]
exp = np.delete(exp, 0, axis=0)
exp = np.delete(exp, 0, axis=1)
exp = exp.astype(np.float64)
exp_row, exp_column = exp.shape

print("Reading exp_array completed.")

adj1 = pd.read_csv(args.pos_adj, header = None)
adj1 = np.array(adj1)
adj1 = np.delete(adj1, 0, axis=0)
adj1 = np.delete(adj1, 0, axis=1)
adj1 = adj1.astype(np.float64)
adj1_row, adj1_column = adj1.shape

print(adj1_row, adj1_column)
print("Reading pos_adj completed.")

dis1 = pd.read_csv(args.pos_dis, header = None)
dis1 = np.array(dis1)
dis1 = dis1.astype(np.float64)
dis1_row, dis1_column = dis1.shape

print("Reading pos_dis completed.")

dis2 = pd.read_csv(args.image_dis, header = None)
dis2 = np.array(dis2)
dis2 = dis2.astype(np.float64)
dis2_row, dis2_column = dis2.shape

print("Reading image_dis completed.")

tmp = np.zeros((exp_row, exp_column))

for i in range(0, adj1_row):
    list1 = np.zeros(exp_column, np.float64)
    nb = 0
    for j in range(0, adj1_column):
        if(adj1[i][j] != 0):
            nb += 1
            cor = 1 / (1 + dis1[i][j])
            for k in range(0, exp_column):
                list1[k] += exp[j][k] * cor
    if (nb > 0):
        for k in range(0, exp_column):
            tmp[i][k] +=  list1[k] * nb

print("Calculating part1 completed.")

for i in range(0, adj1_row):
    list2 = np.zeros(exp_column, np.float64)
    nb = 0
    for j in range(0, adj1_column):
        if(adj1[i][j] != 0):
            nb += 1
            cor = 1 / (1 + dis2[i][j])
            for k in range(0, exp_column):
                list2[k] += exp[j][k] * cor
    if(nb > 0):
        for k in range(0, exp_column):
            tmp[i][k] += list2[k] * nb

for i in range(0, exp_row):
    for j in range(0, exp_column):
        exp[i][j] += tmp[i][j]

print("Calculating part2 completed.")

df = pd.DataFrame(data = exp, index = exp_index, columns = exp_header)
df.to_csv(args.destination, header = True, index = True, mode='a')