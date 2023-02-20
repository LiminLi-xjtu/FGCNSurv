from itertools import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):

    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.f(self.linear[layer](x))
            x = gate * nonlinear + (1-gate) * linear
        return x

class Graph_Survival_Analysis(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.miRNA_encoding = nn.Sequential(nn.Linear(600, 40),
                                            nn.Tanh(),
                                            Highway(40, 5, f=F.relu),
                                            )

        self.RNASeq_encoding = nn.Sequential(nn.Linear(6000, 40),
                                             nn.Tanh(),
                                             Highway(40, 5, f=F.relu),
                                             )

        self.RNASeq_linear = nn.Linear(40, 20, bias=False)
        self.miRNA_linear = nn.Linear(40, 20, bias=False)

        self.feature_fusion = nn.Sequential(nn.BatchNorm1d(60),  nn.Linear(60, 40),  nn.Tanh())
        self.k = 5
        self.GCN_Cox = nn.Sequential(nn.Dropout(0.3),  nn.Linear(40, 40),  nn.ReLU(), nn.Linear(40, 1))

    def get_RNASeq_feature(self, gene):

        gene = gene.view(gene.shape[0], -1)
        # gene = F.tanh(self.fcg(gene))
        # gene = self.bn1_fcg(gene)
        # gene = self.fcg_highway(gene)
        # # gene = F.dropout(gene, 0.3)
        # # gene =F.sigmoid(self.bn2_fcg(gene))
        # return gene
        gene_encoding = self.RNASeq_encoding(gene)
        return gene_encoding

    def get_miRNA_feature(self, miRNA):

        miRNA = miRNA.view(miRNA.shape[0], -1)
        # microRNA = F.tanh(self.fcm(microRNA))
        # microRNA = self.bn1_fcm(microRNA)
        # microRNA = self.fcm_highway(microRNA)
        # # microRNA  = F.dropout(microRNA, 0.3)
        # # microRNA_feature =F.sigmoid(self.bn2_fcm(microRNA))
        # return microRNA
        miRNA_encoding = self.miRNA_encoding(miRNA)
        return miRNA_encoding

    def get_graph(self, RNASeq_feature, miRNA_feature, fbm_feature):

        RNASeq_feature = F.normalize(RNASeq_feature)
        miRNA_feature = F.normalize(miRNA_feature)
        fbm_feature = F.normalize(fbm_feature)
        eps = np.finfo(float).eps
        # Gene
        RNA_emb1 = torch.unsqueeze(RNASeq_feature, 1)  # N*1*d
        RNA_emb2 = torch.unsqueeze(RNASeq_feature, 0)  # 1*N*d
        W_Gene = ((RNA_emb1 - RNA_emb2) ** 2).sum(2)  # N*N*d -> N*N
        W_Gene = torch.exp(-W_Gene / (2*10))
        # keep top-k values
        if self.k > 0:
            topk, indices = torch.topk(W_Gene, 5)
            mask = torch.zeros_like(W_Gene)
            mask = mask.scatter(1, indices, 1)
            # mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W_Gene = W_Gene * mask
        W_Gene = W_Gene / W_Gene.sum(0)
        # fbm
        fbm_emb1 = torch.unsqueeze(fbm_feature, 1)  # N*1*d
        fbm_emb2 = torch.unsqueeze(fbm_feature, 0)  # 1*N*d
        W_fbm = ((fbm_emb1 - fbm_emb2) ** 2).sum(2)  # N*N*d -> N*N
        # print("FMB的特征:")
        # print(W_fbm[10,:])
        W_fbm = torch.exp(-W_fbm /(2*10))
        # keep top-k values
        if self.k > 0:
            topk, indices = torch.topk(W_fbm, 3)
            print(indices[10, :])
            mask = torch.zeros_like(W_fbm)
            mask = mask.scatter(1, indices, 1)
            # mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W_fbm = W_fbm * mask
            # print(W_fbm[1,:])
        W_fbm = W_fbm / W_fbm.sum(0)
        # microRNA
        microRNA_emb1 = torch.unsqueeze(miRNA_feature, 1)  # N*1*d
        microRNA_emb2 = torch.unsqueeze(miRNA_feature, 0)  # 1*N*d
        W_microRNA = ((microRNA_emb1 - microRNA_emb2) ** 2).sum(2)  # N*N*d -> N*N

        W_microRNA = torch.exp(-W_microRNA / (2*10))
        # keep top-k values
        if self.k > 0:
            topk, indices = torch.topk(W_microRNA, 3)
            mask = torch.zeros_like(W_microRNA)
            mask = mask.scatter(1, indices, 1)
            # mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W_microRNA = W_microRNA * mask
        W_microRNA = W_microRNA / W_microRNA.sum(0)

        W = (W_Gene + W_microRNA + W_fbm) + torch.eye(RNASeq_feature.shape[0])
        # print(W[1,:])
        D = torch.sum(W, dim=1)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        S = D_sqrt_inv * W * D_sqrt_inv
        return S

    def get_fbn_feature(self, RNASeq_feature, miRNA_feature):
        # print(RNASeq_feature)
        return torch.sum(torch.reshape(self.RNASeq_linear(RNASeq_feature), (-1, 1, 20)) * torch.reshape(
            self.miRNA_linear(miRNA_feature), (-1, 1, 20)), dim=1)
        # temp_fea = torch.sum(torch.reshape(self.gene_linear(gene_feature), (-1, 1, 20)) *
        #                      torch.reshape(self.microRNA_linear(microRNA_feature), (-1, 1, 20)), dim=1)
        # return F.normalize(torch.sqrt(F.relu(temp_fea)) - torch.sqrt(F.relu(-temp_fea)))


    def get_survival_result(self, gene, miRNA):
        RNASeq_feature = self.get_RNASeq_feature(gene)
        miRNA_feature = self.get_miRNA_feature(miRNA)
        fbn_feature = self.get_fbn_feature(RNASeq_feature, miRNA_feature)
        S = self.get_graph(RNASeq_feature, miRNA_feature, torch.cat((RNASeq_feature + miRNA_feature, fbn_feature), 1))
        X = self.feature_fusion(torch.cat((RNASeq_feature + miRNA_feature, fbn_feature), 1))
        print(self.GCN_Cox(torch.mm(S, X)))
        return self.GCN_Cox(torch.mm(S, X))