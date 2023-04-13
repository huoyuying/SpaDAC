import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import entropy, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score
MI = lambda x, y: mutual_info_score(x, y)
NMI = lambda x, y: normalized_mutual_info_score(x, y, average_method='arithmetic')
AMI = lambda x, y: adjusted_mutual_info_score(x, y, average_method='arithmetic')

def clustering_refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    print('--------------------refine_dis--------------------')
    # print(dis)
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    print('--------------------refine_pred--------------------')
    # print(pred)
    dis_df = pd.DataFrame(dis.values, index=sample_id, columns=sample_id)
    print('--------------------refine_dis_df--------------------')
    # print(dis_df)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for  ST data.")

    for k in range(10):
        refined_pred = []
        for i in range(len(sample_id)):
            index = sample_id[i]

            dis_tmp = dis_df.loc[index, :].sort_values()
            nbs = dis_tmp[0:num_nbs + 1]
            nbs_pred = pred.loc[nbs.index, "pred"]
            self_pred = pred.loc[index, "pred"]
            v_c = nbs_pred.value_counts()
            if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) >= num_nbs / 2):
                refined_pred.append(v_c.idxmax())
            else:
                refined_pred.append(self_pred)
        pred = []
        pred = pd.DataFrame({"pred": refined_pred}, index=sample_id)
    return refined_pred

def res_search_fixed_clus_1(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 5.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

def res_search_fixed_clus_2(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.louvain(adata, random_state=0, resolution=res)
        count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
        if count_unique_louvain == fixed_clus_count:
            break
    return res

def clustering(DLPFC,emb_PATH,n_clusters,n_neighbors=15):
    adata = sc.read_csv(emb_PATH)
    adata.obs["ground_truth"] = DLPFC.obs["ground_truth"].values
    sc.tl.pca(adata)
    sc.pp.neighbors(adata,n_neighbors=n_neighbors,n_pcs=30)
    sc.tl.umap(adata)
    sc.tl.tsne(adata)
    eval_resolution1 = res_search_fixed_clus_1(adata, n_clusters)
    eval_resolution2 = res_search_fixed_clus_2(adata, n_clusters)
    sc.tl.leiden(adata,resolution=eval_resolution1)
    sc.tl.louvain(adata, resolution=eval_resolution2)
    
    DLPFC.obs["leiden"] = adata.obs["leiden"].values
    DLPFC.obs["louvain"] = adata.obs["louvain"].values
    
    ##------------------------------ kmeans-----------------------------
    X = pd.read_csv(emb_PATH,header=None)
    kmeans = KMeans(n_clusters = n_clusters, max_iter = 300, n_init = 10, init = 'k-means++', random_state = 0)
    y_kmeans = kmeans.fit_predict(X)
    new = [str(x) for x in y_kmeans]
    adata.obs["kmeans"] = new
    DLPFC.obs["kmeans"] = adata.obs["kmeans"].values
   



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='clustering',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='151673')
    parser.add_argument("--adj", type=str, default="adj1")
    parser.add_argument("--img", type=str, default="adj2")
    parser.add_argument('--epoch', type=int, default=49)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--n_clusters', default=20, type=int)

    args = parser.parse_args()
    
    ## make adata
    adata = sc.read_visium(f"./{args.name}")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    ground_truth= pd.read_csv(f"../datasets/{args.name}/d{args.name}_groundtruth.csv")
    adata.obs["ground_truth"] = ground_truth["label"].values
    adata.var_names_make_unique()
    
    ## read embedding
    emb_PATH = Path(f"./result/{args.name}/d{args.name}_{args.adj}_{args.img}_{args.epoch}_{args.max_epoch}.csv")
    
    clustering(adata,emb_PATH1,args.n_clusters)
    adj_PATH = Path(f"../datasets/{args.name}/notpca/d{args.name}_def.csv") ## x,y 坐标的欧式距离矩阵
    clustering_refine(adata,emb_PATH,adj_PATH)


