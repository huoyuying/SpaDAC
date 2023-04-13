import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import RMSprop

import utils
from model import GAT
from evaluation import eva
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt


def pretrain():
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.name != "151507" and args.name != "151508" and args.name != "151509" and args.name != "151510" and args.name != "151669" and args.name != "151670" and args.name != "151671" and args.name != "151672" and args.name != "151673" and args.name != "151674" and args.name != "151675" and args.name != "151676" :
        print(f"Using {args.name} dataset")
        adj = pd.read_csv(f"./dlpfc/{args.name}/notpca/{args.name}_{args.adj}.csv",index_col=0)  
        x = pd.read_csv(f"./dlpfc/{args.name}/{args.name}_exp_{args.exp}.csv",index_col=0)
        label= pd.read_csv(f"./dlpfc/{args.name}/{args.name}_groundtruth.csv")
    else : 
        # data process
        print(f"Using DLPFC{args.name} dataset")
        adj = pd.read_csv(f"./dlpfc/{args.name}/notpca/d{args.name}_{args.adj}.csv",index_col=0)  
        x = pd.read_csv(f"./dlpfc/{args.name}/d{args.name}_exp_{args.exp}.csv",index_col=0)
        # N, D = x.shape
        label= pd.read_csv(f"./dlpfc/{args.name}/d{args.name}_groundtruth.csv")
    
    #label["label"] = label["label"].float_id.astype('str')
    #print("change sucessful!!!")
    y = label["label"]
    
    x = torch.Tensor(np.array(x))
    adj = torch.Tensor(np.array(adj))
    adj_label = adj
    print(x.shape[0])
    
    adj += torch.eye(x.shape[0])
    adj = normalize(adj, norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)

    M = utils.get_M(adj).to(device)
    
    xlab = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    
    for epoch in range(args.max_epoch):
        x = x.to(device)
        adj = adj.to(device)
		
        model.train()
        A_pred, z = model(x, adj, M)

        loss = F.binary_cross_entropy(A_pred.contiguous().view(-1).to(device), adj_label.contiguous().view(-1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        with torch.no_grad():
            _, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
            xlab.append(epoch) 
            y1.append(acc)
            y2.append(nmi)
            y3.append(ari)
            y4.append(f1)
            
            
        #if epoch  == args.max_epoch -1:
        if epoch != args.max_epoch:
            torch.save(
                model.state_dict(), f"./pretrain/pretrain_double/predaegc_{args.name}_{args.exp}_{args.adj}_{epoch}.pkl"
            )
    b = z.to(device).data.cpu().numpy()
	#b = z.numpy()
    #print(b)

    # np.savetxt(f"./result/{args.name}/{args.name}_{args.exp}_{args.adj}_{args.max_epoch}_{args.lr}_{args.embedding_size}.csv",b)
    
    plt.figure()
    plt.plot(xlab, y1, 'red', marker='s',label='acc',ms=1)
    plt.plot(xlab, y2, 'green', marker='o',label='nmi',ms=1)
    plt.plot(xlab, y3, 'blue', marker='^',label='ari',ms=1)
    plt.plot(xlab, y4, 'purple', marker='+',label='f1',ms=1)
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('index(0-1)')
    # plt.xlim((0,50))
    plt.ylim((0,1))
    # plt.title(f'{args.name}_{args.adj}_{args.exp}_{args.max_epoch}_{args.lr}_{args.embedding_size}')
    # plt.savefig(f"./result/{args.name}_{args.adj}_{args.exp}_{args.max_epoch}_{args.lr}_{args.embedding_size}.png")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="151673")
    parser.add_argument("--adj", type=str, default="adj1")
    parser.add_argument("--exp", type=int, default=3000)    
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=7, type=int)
    parser.add_argument("--hidden_size", default=1024, type=int)# 256
    parser.add_argument("--embedding_size", default=256, type=int)# 16
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda:3" if args.cuda else "cpu")

    # datasets = utils.get_dataset(args.name)
    
    if args.name == "151672" or args.name == "151671" :
        args.n_clusters = 5
        args.lr = 0.0005
    elif args.name == "151670" or args.name == "151669":
        args.n_clusters = 5
    elif args.name == "section1":
        args.n_clusters = 20
    elif args.name == "PDAC":
        args.n_clusters = 4
    elif args.name == "MERFISH1":
        args.n_clusters = 5
    elif args.name == "merfish0.26":
        args.exp = 160
        args.n_clusters = 16
        args.hidden_size = 128
        args.embedding_size = 32
        args.lr = 0.01
    elif args.name == "merfish0.21":
        args.exp = 160
        args.n_clusters = 16
        args.hidden_size = 128
        args.embedding_size = 32
        args.lr = 0.01
    elif args.name == "s1_an":
        args.n_clusters = 12
    elif args.name == "s1_pos":
        args.n_clusters = 11
    elif args.name == "s2_an":
        args.n_clusters = 10        
    elif args.name == "s2_pos":
        args.n_clusters = 10
    elif args.name == "stereoseq1":
        args.n_clusters = 10
    elif args.name == "stereoseq2": 
        args.n_clusters = 12
    elif args.name == "slideseq":
        args.n_clusters = 10
        args.lr = 0.01
    elif args.name == "slideseqv2":
        args.n_clusters = 14
    elif args.name == "slideseqv2_data2":
        args.n_clusters = 11
    else :
        print("please enter cluster number!!!!")

    # args.input_dim = dataset.num_features
    args.input_dim = args.exp


    print(args)
    # pretrain(dataset)
    pretrain()
    
