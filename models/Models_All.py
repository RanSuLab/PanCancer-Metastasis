import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch.utils.data as data_utils
import torch.optim as optim
from torch.nn import Linear
from torchnet import meter as tnt  # 计算混淆矩阵

from torch_geometric.nn import global_mean_pool,global_add_pool



class PyG_GCN(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nclass, dropout):
        super(PyG_GCN, self).__init__()
        torch.manual_seed(12345) 
        self.conv1 = GCNConv(nfeat, nhid_1)
        self.conv2 = GCNConv(nhid_1, nclass)
        self.dropout = dropout
        self.lin1 = Linear(nclass, 64)
        self.lin2 = Linear(64, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = global_add_pool(x, batch, size=None)
        x = self.lin1(x)
        x = x.relu()
        out = self.lin2(x)
        Y_prob = F.log_softmax(out, dim=1)  
        Y_hat = Y_prob.max(1)[1].float()  
        return Y_prob, Y_hat, out

    def calculate_objective(self, node_features, bag_label, edge_index, edge_weight, batch):
        bag_label = bag_label.long()
        Y_prob, Y_hat, A = self.forward(node_features, edge_index, edge_weight, batch)
        loss = F.nll_loss(Y_prob, bag_label)
        return loss, A

    def calculate_classification_error(self, node_features, bag_label, edge_index, edge_weight, batch):
        bag_label = bag_label.float()
        Y_prob, Y_hat, _ = self.forward(node_features, edge_index, edge_weight, batch)
        classer = tnt.ClassErrorMeter(topk=[1], accuracy=True)
        Y_prob = Y_prob.detach().cpu().numpy()
        bag_label = bag_label.detach().cpu().numpy()
        classer.add(Y_prob, bag_label)
        right = classer.value()[0] / 100
        return right, Y_hat,Y_prob


class GraphSAGE_Net(torch.nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nclass, dropout):
        super(GraphSAGE_Net, self).__init__()
        self.sage1 = SAGEConv(nfeat, nhid_1)
        self.sage2 = SAGEConv(nhid_1, nclass)
        self.dropout = dropout
        self.lin1 = Linear(nclass, 64)
        self.lin2 = Linear(64, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.sage1(x, edge_index)
        x = x.relu()               
        x = self.sage2(x, edge_index)
        x = x.relu()
        x = global_add_pool(x, batch, size=None) 
        x = self.lin1(x)         
        x = x.relu()
        out = self.lin2(x)     
        Y_prob = F.log_softmax(out, dim=1)
        Y_hat = Y_prob.max(1)[1].float()
        return Y_prob, Y_hat, out

    def calculate_objective(self, node_features, bag_label, edge_index, edge_weight, batch):
        bag_label = bag_label.long()
        Y_prob, Y_hat, A = self.forward(node_features, edge_index, edge_weight, batch)
        loss = F.nll_loss(Y_prob, bag_label)

        return loss, A

    def calculate_classification_error(self, node_features, bag_label, edge_index, edge_weight, batch):

        bag_label = bag_label.float()
        Y_prob, Y_hat, _ = self.forward(node_features, edge_index, edge_weight, batch)
        classer = tnt.ClassErrorMeter(topk=[1], accuracy=True)
        Y_prob = Y_prob.detach().cpu().numpy()
        bag_label = bag_label.detach().cpu().numpy()
        classer.add(Y_prob, bag_label)
        right = classer.value()[0] / 100
        return right, Y_hat,Y_prob


class GAT_Net(torch.nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nclass, dropout, heads=2):
        super(GAT_Net, self).__init__()
        self.gat1 = GATConv(nfeat, nhid_1, heads=heads)
        self.gat2 = GATConv(nhid_1 * heads, nclass)
        self.dropout = dropout
        self.lin1 = Linear(nclass,64)
        self.lin2 = Linear(64, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.gat1(x, edge_index)
        x = x.relu()
        x = self.gat2(x, edge_index)
        x = x.relu()
        x = global_add_pool(x, batch, size=None)
        x = self.lin1(x) 
        x = x.relu()
        out = self.lin2(x) 

        Y_prob = F.log_softmax(out, dim=1)
        Y_hat = Y_prob.max(1)[1].float()
        return Y_prob, Y_hat, out

    def calculate_objective(self, node_features, bag_label, edge_index, edge_weight, batch):
        bag_label = bag_label.long()
        Y_prob, Y_hat, A = self.forward(node_features, edge_index, edge_weight, batch)
        loss = F.nll_loss(Y_prob, bag_label)
        return loss, A

    def calculate_classification_error(self, node_features, bag_label, edge_index, edge_weight, batch):
        bag_label = bag_label.float()
        Y_prob, Y_hat, _ = self.forward(node_features, edge_index, edge_weight, batch)
        classer = tnt.ClassErrorMeter(topk=[1], accuracy=True)
        Y_prob = Y_prob.detach().cpu().numpy()
        bag_label = bag_label.detach().cpu().numpy()
        classer.add(Y_prob, bag_label)
        right = classer.value()[0] / 100
        return right, Y_hat,Y_prob
