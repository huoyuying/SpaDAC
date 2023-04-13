import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layer import GATLayer


class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M1, img, M2):
        h1 = self.conv1(x, adj, M1)
        h2 = self.conv2(h1, adj, M1)
        z1 = F.normalize(h2, p=2, dim=1)
        # print(z1)
            
        h3 = self.conv1(x, img, M2)
        h4 = self.conv2(h3, img, M2)
        z2 = F.normalize(h4, p=2, dim=1)
        # print(z2)
        
        z = torch.cat((z1,z2),1)
        # z = np.concatenate(z1,z2)
        # print(z)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
