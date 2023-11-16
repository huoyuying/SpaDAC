import argparse
import numpy
import numpy as np
import warnings
import pandas as pd
import time
import os
warnings.filterwarnings("ignore")

def compute_matrix(adj,num):
	n = adj.shape[0]
	matrix = np.empty((n+1, n+1), dtype=np.float32)
	raw = adj.copy()
	for i in range(n):
		limit = np.sort(raw[i,])[num]
		print(limit)
		for j in range(n):
			if adj[i,j] <= limit and i!=j:
				matrix[i+1,j+1] = 1
			else:
				matrix[i+1,j+1] = 0
	print(matrix)
	return matrix

def adj_euclidean(emb):
	n = emb.shape[0]
	print(n)
	print("*"*50)
	adj = np.empty((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(i, n):
			adj[i][j] = adj[j][i] = numpy.sqrt(numpy.sum(numpy.square(emb[i]-emb[j])))
	print("*" * 50)
	print("euclidean:")
	print(adj)
	return adj

def adj_cosine(emb):
	n = emb.shape[0]
	print(n)
	print("*" * 50)
	adj = np.empty((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(i, n):
			adj[i][j] = adj[j][i] = np.dot(emb[i],emb[j])/(np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]))
	print("*" * 50)
	print("cosine:")
	print(adj)
	return adj

def adj_pearson(emb):
	n = emb.shape[0]
	print(n)
	print("*" * 50)
	adj = np.empty((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(i, n):
			i_ = emb[i] - np.mean(i)
			j_ = emb[j] - np.mean(j)
			adj[i][j] = adj[j][i] = np.dot(i_, j_) / (np.linalg.norm(i_) * np.linalg.norm(j_))
	print("*" * 50)
	print("pearson:")
	print(adj)
	return adj

def adj_jaccard(emb):
	n = emb.shape[0]
	print(n)
	print("*" * 50)
	adj = np.empty((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(n):
			i = np.asarray(emb[i], np.int32)
			j = np.asarray(emb[j], np.int32)
			up = np.double(np.bitwise_and((i!=j),np.bitwise_or(i!=0,j!=0)).sum())
			down = np.double(np.bitwise_or(i!=0,j!=0).sum())
			adj[i][j] = up/down
	print("*" * 50)
	print("jaccard:")
	print(adj)
	return adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, default="151673")
    parser.add_argument("--model", type=str, default="inception_v3")
    parser.add_argument("--distance", type=str, default="euc") # euclidean cosine pearson jaccard 
    parser.add_argument("--neighbor",type=int,default=6)
    args = parser.parse_args()
   
    start = time.time() 
    #emb = np.load(f"./cut_image/{args.name}/emb_{args.name}_{args.model}.npy")
    emb = np.load(f"./cut_image/test_time/emb_{args.name}_{args.model}.npy")
    print(emb.shape)

    # model_pca = PCA(n_components=50)
    # emb = model_pca.fit_transform(emb)
    # print(emb.shape)

    if args.distance == "euc":
        adj = adj_euclidean(emb)
    elif args.distance == "cos":
        adj = adj_cosine(emb)
    elif args.distance == "pea":
        adj = adj_pearson(emb)
    elif args.distance =="jac":
        adj = adj_jaccard(emb)
    else:
        adj = adj_euclidean(emb)
    
   # os.makedirs(f'./cut_image/{args.name}/cor_adj')
    print("mkdir sucessful!")
    #np.savetxt(f'./cut_image/{args.name}/cor_adj/cor_{args.name}_{args.model}_{args.distance}.csv', adj, delimiter=',') 
    matrix = compute_matrix(adj,args.neighbor)
    np.savetxt(f'./cut_image/test_time/cor_{args.name}_{args.model}_{args.distance}.csv', adj, delimiter=',')
    np.savetxt(f'./cut_image/test_time/adj_{args.name}_{args.model}_{args.distance}.csv', matrix, delimiter=',')
    #np.savetxt(f'./cut_image/{args.name}/cor_adj/adj_{args.name}_{args.model}_{args.distance}.csv', matrix, delimiter=',')
    
    print("save sucessful!!")
    end = time.time()
    print("run time: "+ str(end-start) + " seconds")

