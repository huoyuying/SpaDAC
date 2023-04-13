import argparse
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import RMSprop

import os

import utils
from model_plus import GAT
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
        img = pd.read_csv(f"./dlpfc/{args.name}/notpca/{args.name}_{args.img}.csv",index_col=0)        
        x = pd.read_csv(f"./dlpfc/{args.name}/{args.name}_exp_{args.exp}.csv",index_col=0)
        label= pd.read_csv(f"./dlpfc/{args.name}/{args.name}_groundtruth.csv")
    else : 
        # data process
        print(f"Using DLPFC{args.name} dataset")
        adj = pd.read_csv(f"./dlpfc/{args.name}/notpca/d{args.name}_{args.adj}.csv",index_col=0)  
        img = pd.read_csv(f"./dlpfc/{args.name}/notpca/d{args.name}_{args.img}.csv",index_col=0)
        x = pd.read_csv(f"./dlpfc/{args.name}/d{args.name}_exp_{args.exp}.csv",index_col=0)
        # N, D = x.shape
        label= pd.read_csv(f"./dlpfc/{args.name}/d{args.name}_groundtruth.csv")
        
    y = label["label"]
    
    x = torch.Tensor(np.array(x))
    adj = torch.Tensor(np.array(adj))
    adj_label = adj   
    adj += torch.eye(x.shape[0])
    adj = normalize(adj, norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)
    M1 = utils.get_M(adj).to(device)
    
    img = torch.Tensor(np.array(img))  
    img += torch.eye(x.shape[0])
    img = normalize(img, norm="l1")
    img = torch.from_numpy(img).to(dtype=torch.float)
    M2 = utils.get_M(img).to(device)
    
    xlab = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    
    # x = x.to((device)
    # adj = adj.to(device)
    # img = img.to(device)
    #os.makedirs(f'./pretrain/{args.name}_{args.lr}')
    #print("mkdir sucessful!")
    for epoch in range(args.max_epoch):
        x = x.to(device)
        adj = adj.to(device)
        img = img.to(device)
        model.train()
      
        A_pred, z = model(x, adj, M1, img, M2)

        loss = F.binary_cross_entropy(A_pred.contiguous().view(-1).to(device), adj_label.contiguous().view(-1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        with torch.no_grad():
            _, z = model(x, adj, M1, img, M2)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
            xlab.append(epoch) 
            y1.append(acc)
            y2.append(nmi)
            y3.append(ari)
            y4.append(f1)
            
            
      #  if epoch != args.max_epoch - 1:
        if epoch != 100 :  
            torch.save(
                model.state_dict(), f"./pretrain/test_time/predaegc_{args.name}_{args.exp}_{args.adj}_{args.img}_{epoch}.pkl"
            )
    b = z.to(device).data.cpu().numpy()
    # print(b)

    # np.savetxt(f"./result/{args.name}/{args.name}_{args.adj}_{args.exp}_{args.max_epoch}_{args.lr}_{args.embedding_size}.csv",b)
    
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
    plt.title(f'{args.name}_{args.adj}_{args.exp}_{args.max_epoch}')
    # plt.savefig(f"./result/{args.name}/{args.name}_{args.adj}_{args.exp}_{args.max_epoch}_{args.lr}_{args.embedding_size}.png")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="151673")
    parser.add_argument("--adj", type=str, default="adj1")
    parser.add_argument("--img", type=str, default="adj2")
    parser.add_argument("--exp", type=int, default=2000)    
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
    print(device) 
    # datasets = utils.get_dataset(args.name)
    
    if args.name == "151669" or args.name == "151670":
        args.n_clusters = 5
    elif args.name == "151671" or args.name == "151672":
        args.n_clusters = 5
        args.lr = 0.0005
    elif args.name == "section1":
        args.n_clusters = 20
    elif args.name == "PDAC":
        args.n_clusters = 4
    elif args.name == "s1_an" or args.name == "s2_an":
        args.n_clusters = 15 # 13 #12
        args.lr = 0.0001
    elif args.name == "s1_pos" or args.name == "s2_pos":
        args.n_clusters = 20 #17 #16 #10
        args.lr = 0.0001
    else :
        args.n_clusters = 7
    
    start = time.time()
    # args.input_dim = dataset.num_features
    args.input_dim = args.exp

    print(args)
    
    pretrain()
    end = time.time()
    print("run time: "+ str(end-start) + " seconds")



