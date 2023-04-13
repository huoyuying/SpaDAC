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
import os

from torch_geometric.datasets import Planetoid

import utils
from model import GAT
from evaluation import eva
import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '5678'

torch.distributed.init_process_group(backend="nccl")
#local_rank = torch.distributed.get_rank()

class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size,embedding_size, alpha)
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def trainer(local_rank):
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)

    #print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    if args.name != "151507" and args.name != "151508" and args.name != "151509" and args.name != "151510" and args.name != "151669" and args.name != "151670" and args.name != "151671" and args.name != "151672" and args.name != "151673" and args.name != "151674" and args.name != "151675" and args.name != "151676" :
        print(f"Using {args.name} dataset")
        adj = pd.read_csv(f"./dlpfc/{args.name}/notpca/{args.name}_{args.adj}.csv",index_col=0)  
        x = pd.read_csv(f"./dlpfc/{args.name}/{args.name}_exp_{args.exp}.csv",index_col=0)
        label= pd.read_csv(f"./dlpfc/{args.name}/{args.name}_groundtruth.csv")
    else : 
        # data process
        print(f"Using dlpfc{args.name} dataset")
        adj = pd.read_csv(f"./dlpfc/{args.name}/notpca/d{args.name}_{args.adj}.csv",index_col=0)  
        x = pd.read_csv(f"./dlpfc/{args.name}/d{args.name}_exp_{args.exp}.csv",index_col=0)
        # N, D = x.shape
        label= pd.read_csv(f"./dlpfc/{args.name}/d{args.name}_groundtruth.csv")
    
    y = label["label"]
    
    data = torch.Tensor(np.array(x))
    adj = torch.Tensor(np.array(adj))
    adj_label = adj
    
    adj += torch.eye(data.shape[0])
    adj = normalize(adj, norm="l1")
    adj = torch.from_numpy(adj).to(dtype=torch.float)

    M = utils.get_M(adj).to(device)
 
    with torch.no_grad():
        data = data.to(device)
        adj = adj.to(device)
        #_, z = model.gat(data, adj, M)
        _, z = model.module.gat(data, adj, M)
    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.module.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pretrain')

    for epoch in range(args.max_epoch):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, Q = model(data, adj, M)
            
            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            list = []
            list.append(eva(y,q,epoch))
			# eva(y, q, epoch)

        A_pred, z, q = model(data, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.contiguous().view(-1).to(device), adj_label.contiguous().view(-1).to(device))

        # loss = 10 * kl_loss + re_loss
        loss = 10 * kl_loss + re_loss
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #b = z.detach().numpy()
    #print(b)

    #np.savetxt(f"./result/{args.name}/d{args.name}_{args.adj}_{args.exp}_{args.epoch}_{args.max_epoch}.csv",b)

    b = z.to(device).data.cpu().detach().numpy()
    # print(b)
    print(list)
    colname = ['acc', 'nmi', 'ari', 'f1']
    test = pd.DataFrame(columns=colname,data=list)
    test.to_csv(f"./result/double/{args.name}/d{args.name}_{args.adj}_{args.epoch}_{args.max_epoch}_ari.txt")
    np.savetxt(f"./result/double/{args.name}/d{args.name}_{args.adj}_{args.epoch}_{args.max_epoch}.csv",b)

if __name__ == "__main__":
    print("=====hanmh")
    parser = argparse.ArgumentParser(description='train',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='151673')
    parser.add_argument("--adj", type=str, default="adj1")
    parser.add_argument("--exp", type=int, default=3000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument('--epoch', type=int, default=49)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=1024, type=int) # 256
    parser.add_argument('--embedding_size', default=256, type=int) # 16
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--local_rank', type=int, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    #device = torch.device("cuda:3" if args.cuda else "cpu")
    print(f"local_rank: {args.local_rank}")
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda", args.local_rank)
    
    # datasets = utils.get_dataset(args.name)
    # dataset = datasets[0]

    if args.name == "151669" or args.name == "151670" or args.name == "151671" or args.name == "151672" :
        args.n_clusters = 5
        args.lr = 0.0005
    elif args.name == "151671" or args.name == "151672":
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
        args.lr = 0.0005
    elif args.name == "merfish0.21":
        args.exp = 160
        args.n_clusters = 16
        args.hidden_size = 128
        args.embedding_size = 32
        args.lr = 0.0005
    elif args.name == "s1_an":
        args.n_clusters = 12
    elif args.name == "s1_pos":
        args.n_clusters = 11
    elif args.name == "s2_an":
        args.n_clusters = 10        
    elif args.name == "s2_pos":
        args.n_clusters = 10
    elif args.name == "slideseq":
        args.n_clusters = 10
    elif args.name == "stereoseq1":
        args.n_clusters = 10
    elif args.name == "stereoseq2":
        args.n_clusters = 12
    elif args.name == "slideseqv2":
        args.n_clusters = 14
    elif args.name == "slideseqv2_data2":
        args.n_clusters = 11
    else :
        print("please enter cluster number!!!!")
    
    
    args.pretrain_path = f'./pretrain/pretrain_double/predaegc_{args.name}_{args.exp}_{args.adj}_{args.epoch}.pkl'
    # args.input_dim = dataset.num_features
    args.input_dim = args.exp


    #print(args)
    #trainer(dataset)
    trainer(args.local_rank)
