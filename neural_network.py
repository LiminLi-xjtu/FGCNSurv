from itertools import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing


class Highway(nn.Module):

    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class Graph_Survival_Analysis(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.miRNA_encoding = nn.Sequential(nn.Linear(600, 50),
                                            nn.Tanh(),
                                            nn.BatchNorm1d(50),
                                            Highway(50, 3, f=F.relu),
                                            )

        self.RNASeq_encoding = nn.Sequential(nn.Linear(6000, 50),
                                             nn.Tanh(),
                                             nn.BatchNorm1d(50),
                                             Highway(50, 3, f=F.relu),
                                             )

        self.RNASeq_linear = nn.Linear(50, 200, bias=False)
        self.miRNA_linear = nn.Linear(50, 200, bias=False)
        self.fbm_bn = nn.BatchNorm1d(10)
        self.plus_bn = nn.BatchNorm1d(50)
        self.dropout = nn.Dropout(0.3)
        self.GCN_feature_fusion = nn.Sequential(nn.Linear(60, 60), nn.Tanh(), nn.Linear(60, 30), nn.Sigmoid())
        self.Cox = nn.Sequential(nn.Linear(30, 1))

    def get_RNASeq_feature(self, gene):

        gene = gene.view(gene.shape[0], -1)
        gene_encoding = self.RNASeq_encoding(gene)
        # print("RNASeq feature:")
        # print(gene_encoding)
        return gene_encoding

    def get_miRNA_feature(self, miRNA):

        miRNA = miRNA.view(miRNA.shape[0], -1)
        miRNA_encoding = self.miRNA_encoding(miRNA)
        # print("miRNA feature:")
        # print(miRNA_encoding)
        return miRNA_encoding


    def get_fbn_feature(self, RNASeq_feature, miRNA_feature):
        temp_fea = torch.sum(torch.reshape(self.RNASeq_linear(RNASeq_feature), (-1, 20, 10)) *
                             torch.reshape(self.miRNA_linear(miRNA_feature), (-1, 20, 10)), dim=1)
        return torch.sqrt(F.relu(temp_fea)) - torch.sqrt(F.relu(-temp_fea))

    def get_survival_result(self, gene, miRNA, S):
        RNASeq_feature = self.get_RNASeq_feature(gene)
        miRNA_feature = self.get_miRNA_feature(miRNA)
        fbn_feature = self.get_fbn_feature(RNASeq_feature, miRNA_feature)
        X = torch.cat((F.normalize((RNASeq_feature +miRNA_feature)),  fbn_feature), 1)

        return self.Cox(self.GCN_feature_fusion(self.dropout(torch.cat((self.plus_bn(torch.mm(S, X)[:, 0:50]),self.fbm_bn(torch.mm(S, X)[:, 50:60])), 1))))
