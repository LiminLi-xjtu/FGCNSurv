import pandas as pd
from sklearn.model_selection import KFold
import sys

sys.path.append("..")
import numpy as np
import torch
import random
import math
# import argparse
# import json
from neural_network import Graph_Survival_Analysis
# import torch.utils.data as Data
from sklearn import preprocessing




def CIndex(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    concord = 0.
    total = 0.
    eav_count = 0
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        concord = concord + 1
                    elif theta[j] == theta[i]:
                        concord = concord + 0.5
                        eav_count = eav_count + 1
                        print("相等的对数")
                        print(eav_count)
    return concord / total


if __name__ == '__main__':

    data_base_path = "../BRCA"
    RNASeq_dataframe = pd.read_csv(data_base_path + "/RNASeq.csv").iloc[:, 0:6000]
    clinical_dataframe = pd.read_csv(data_base_path + "/clinical.csv")
    miRNA_dataframe = pd.read_csv(data_base_path + "/miRNA.csv").iloc[:, 0:600]
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    train_rate = 0.8
    train_lr = 2e-4
    RNASeq_feature = np.array(RNASeq_dataframe)
    miRNA_feature = np.array(miRNA_dataframe)

    clinical_feature = np.array(clinical_dataframe)
    ystatus = np.squeeze(clinical_feature[:, 0])
    ytime = np.squeeze(clinical_feature[:, 1])
    max_cind_list = []
    for k_num in range(20):
        ystatus_dead_index = np.squeeze(np.argwhere(ystatus == 1))
        ystatus_censor_index = np.squeeze(np.argwhere(ystatus == 0))

        ystatus_dead = ystatus[ystatus_dead_index,]
        ystatus_censor = ystatus[ystatus_censor_index,]
        RNASeq_dead = RNASeq_feature[ystatus_dead_index,]
        RNASeq_censor = RNASeq_feature[ystatus_censor_index,]
        miRNA_dead = miRNA_feature[ystatus_dead_index,]
        miRNA_censor = miRNA_feature[ystatus_censor_index,]
        ytime_dead = ytime[ystatus_dead_index,]
        ytime_censor = ytime[ystatus_censor_index,]

        dead_range = range(RNASeq_dead.shape[0])
        ind_dead_train = random.sample(dead_range, math.floor(RNASeq_dead.shape[0] * train_rate))
        RNASeq_dead_train = RNASeq_dead[ind_dead_train,]
        print(RNASeq_dead_train.shape)
        miRNA_dead_train = miRNA_dead[ind_dead_train,]
        ytime_dead_train = ytime_dead[ind_dead_train,]
        ystatus_dead_train = ystatus_dead[ind_dead_train,]

        ind_dead_rest = [i for i in dead_range if i not in ind_dead_train]
        RNASeq_dead_rest = RNASeq_dead[ind_dead_rest,]
        miRNA_dead_rest = miRNA_dead[ind_dead_rest,]
        ytime_dead_rest = ytime_dead[ind_dead_rest,]
        ystatus_dead_rest = ystatus_dead[ind_dead_rest,]

        censor_range = range(RNASeq_censor.shape[0])
        ind_censor_train = random.sample(censor_range, math.floor(RNASeq_censor.shape[0] * train_rate))

        RNASeq_censor_train = RNASeq_censor[ind_censor_train,]
        miRNA_censor_train = miRNA_censor[ind_censor_train,]
        ytime_censor_train = ytime_censor[ind_censor_train,]
        ystatus_censor_train = np.squeeze(ystatus_censor[ind_censor_train,])

        RNASeq_train = np.concatenate((RNASeq_dead_train, RNASeq_censor_train), axis=0)
        miRNA_train = np.concatenate((miRNA_dead_train, miRNA_censor_train), axis=0)
        ystatus_train = np.squeeze(np.concatenate((ystatus_dead_train, ystatus_censor_train), axis=0))
        ytime_train = np.squeeze(np.concatenate((ytime_dead_train, ytime_censor_train), axis=0))

        ind_censor_rest = [i for i in censor_range if i not in ind_censor_train]
        RNASeq_censor_rest = RNASeq_censor[ind_censor_rest,]
        miRNA_censor_rest = miRNA_censor[ind_censor_rest,]
        ytime_censor_rest = np.squeeze(ytime_censor[ind_censor_rest,])
        ystatus_censor_rest = np.squeeze(ystatus_censor[ind_censor_rest,])

        RNASeq_val = np.concatenate((RNASeq_dead_rest, RNASeq_censor_rest), axis=0)
        miRNA_val = np.concatenate((miRNA_dead_rest, miRNA_censor_rest), axis=0)
        ytime_val = np.squeeze(np.concatenate((ytime_dead_rest, ytime_censor_rest), axis=0))
        ystatus_val = np.squeeze(np.concatenate((ystatus_dead_rest, ystatus_censor_rest), axis=0))

        model = Graph_Survival_Analysis()

        RNASeq_val_tensor = torch.tensor(RNASeq_val, dtype=torch.float)
        miRNA_val_tensor = torch.tensor(miRNA_val, dtype=torch.float)
        RNASeq_train_tensor = torch.tensor(RNASeq_train, dtype=torch.float)
        miRNA_train_tensor = torch.tensor(miRNA_train, dtype=torch.float)

        k = 10
        RNASeq_tensor = torch.cat((RNASeq_train_tensor, RNASeq_val_tensor), axis=0)
        RNASeq_tensor1 = torch.unsqueeze(RNASeq_tensor, 1)  # N*1*d
        RNASeq_tensor2 = torch.unsqueeze(RNASeq_tensor, 0)  # 1*N*d
        W_Gene = ((RNASeq_tensor1 - RNASeq_tensor2) ** 2).sum(2)  # N*N*d -> N*N
        W_Gene_temp =W_Gene.reshape(-1,1)
        distance = torch.median(W_Gene_temp , 0)
        print(distance[0])
        for i in range(W_Gene.shape[0]):
            W_Gene[i, :] = W_Gene[i, :]/(0.3*distance[0])
        W_Gene = torch.exp(-W_Gene)
        if k > 0:
            topk, indices = torch.topk(W_Gene, k)
            mask = torch.zeros_like(W_Gene)
            mask = mask.scatter(1, indices, 1)
            # mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            # mask = ((mask > 0) & (torch.t(mask) > 0)).type(torch.float32)  # intersection, kNN graph
            W_Gene = W_Gene * mask
        # W_Gene = W_Gene / W_Gene.sum(0)
        print(W_Gene)
        miRNA_tensor = torch.cat((miRNA_train_tensor, miRNA_val_tensor), axis=0)
        miRNA_tensor1 = torch.unsqueeze(miRNA_tensor, 1)  # N*1*d
        miRNA_tensor2 = torch.unsqueeze(miRNA_tensor, 0)  # 1*N*d
        W_miRNA = ((miRNA_tensor1 - miRNA_tensor2) ** 2).sum(2)  # N*N*d -> N*N
        W_miRNA_temp =W_miRNA.reshape(-1,1)
        distance = torch.median(W_miRNA_temp , 0)
        print(distance[0])
        for i in range(W_miRNA.shape[0]):
            W_miRNA[i, :] = W_miRNA[i, :]/(0.2*distance[0])
        W_miRNA = torch.exp(-W_miRNA)
        if k > 0:
            topk, indices = torch.topk(W_miRNA, k)
            mask = torch.zeros_like(W_miRNA)
            mask = mask.scatter(1, indices, 1)
            # mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
            # mask = ((mask > 0) & (torch.t(mask) > 0)).type(torch.float32)  # intersection, kNN graph
            W_miRNA = W_miRNA * mask
        # W_miRNA = W_miRNA / W_miRNA .sum(0)
        print(W_miRNA)
        eps = np.finfo(float).eps
        W = (W_Gene + W_miRNA)/2 + torch.eye(W_Gene.shape[0])
        D = torch.sum(W, dim=1)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        S = D_sqrt_inv * W * D_sqrt_inv
        model.eval()

        RNASeq_tensor = torch.cat((RNASeq_train_tensor, RNASeq_val_tensor), 0)
        miRNA_tensor = torch.cat((miRNA_train_tensor, miRNA_val_tensor), 0)
        row = RNASeq_val_tensor.shape[0]
        pred_val = model.get_survival_result(RNASeq_tensor, miRNA_tensor, S)[row:, :]
        cind_val = CIndex(pred_val.detach().numpy(), ytime_val, ystatus_val)
        with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_randomseed' + str(k_num) + '.log', 'a') as f:
            f.writelines('Iteration:' + str(0) + ",cind:" + str(cind_val) + '\n')
        max_cind = 0
        count = 0
        optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=train_lr, weight_decay=5e-4)
        iter = 0

        while True:
            iter = iter + 1
            model.train()
            index = np.squeeze(np.arange(0, RNASeq_train.shape[0]))
            fir_index = np.random.choice(index, size=RNASeq_train.shape[0], replace=False)
            # print(fir_index)
            # seco_index = np.array(list(set(index.tolist()) - set(fir_index)))

            RNASeq_train_tensor = torch.tensor(RNASeq_train, dtype=torch.float)
            miRNA_train_tensor = torch.tensor(miRNA_train, dtype=torch.float)
            RNASeq_tensor = torch.cat((RNASeq_train_tensor, RNASeq_val_tensor), 0)
            miRNA_tensor = torch.cat((miRNA_train_tensor, miRNA_val_tensor), 0)
            ystatus_batch_train = ystatus_train[fir_index,]
            ystatus_train_tensor = torch.tensor(ystatus_batch_train, dtype=torch.float)

            ytime_batch_train = ytime_train[fir_index,]
            ytime_train_tensor = torch.tensor(ytime_batch_train, dtype=torch.float)

            real_batch_size = ystatus_train_tensor.shape[0]
            R_matrix_batch_train = torch.tensor(np.zeros([real_batch_size, real_batch_size], dtype=int),
                                                dtype=torch.float)

            for i in range(real_batch_size):
                R_matrix_batch_train[i,] = torch.tensor(
                    np.array(list(map(int, (ytime_train_tensor >= ytime_train_tensor[i])))))
            model.train()
            theta = model.get_survival_result(RNASeq_tensor, miRNA_tensor, S)[fir_index,]
            exp_theta = torch.reshape(torch.exp(theta), [real_batch_size])
            theta = torch.reshape(theta, [real_batch_size])
            fuse_loss = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta,
                                                                                     R_matrix_batch_train),
                                                                           dim=1))),
                                              torch.reshape(ystatus_train_tensor, [real_batch_size])))

            loss = fuse_loss
            # Update meta-parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # end = time.time()
            # print("1 iteration time:", end - start)


            model.eval()

            ystatus_train_tensor = torch.tensor(ystatus_train, dtype=torch.float)
            ytime_train_tensor = torch.tensor(ytime_train, dtype=torch.float)

            real_batch_size = RNASeq_train_tensor.shape[0]
            R_matrix_batch_train = torch.tensor(np.zeros([real_batch_size, real_batch_size], dtype=int),
                                                dtype=torch.float)
            row = RNASeq_train_tensor.shape[0]
            for i in range(real_batch_size):
                R_matrix_batch_train[i,] = torch.tensor(
                    np.array(list(map(int, (ytime_train_tensor >= ytime_train_tensor[i])))))
            theta = model.get_survival_result(RNASeq_tensor, miRNA_tensor, S)[0:row, :]
            exp_theta = torch.reshape(torch.exp(theta), [real_batch_size])
            theta = torch.reshape(theta, [real_batch_size])
            fuse_loss = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta,
                                                                                     R_matrix_batch_train),
                                                                           dim=1))),
                                              torch.reshape(ystatus_train_tensor, [real_batch_size])))

            print('cox损失:' + str(fuse_loss))
            print("迭代次数：" + str(iter))
            # end = time.time()
            # print("1 iteration time:", end - start)

            cind_train = CIndex(theta, ytime_train, ystatus_train)
            # auc_train = AUC(theta, ytime_train, ystatus_train)

            with open('Cox_Result_lr5e-05/Cox_Train/Regularizer_eval_randomseed' + str(k_num) + '.log',
                      'a') as f:
                f.writelines('Iteration:' + str(iter) + ",cind_train:" + str(cind_train) + '\n')
            RNASeq_val_tensor = torch.tensor(RNASeq_val, dtype=torch.float)
            miRNA_val_tensor = torch.tensor(miRNA_val, dtype=torch.float)
            RNASeq_tensor = torch.cat((RNASeq_train_tensor, RNASeq_val_tensor), 0)
            miRNA_tensor = torch.cat((miRNA_train_tensor, miRNA_val_tensor), 0)
            pred_val = model.get_survival_result(RNASeq_tensor, miRNA_tensor, S)[row:, :]
            cind_val = CIndex(pred_val.detach().numpy(), ytime_val, ystatus_val)
            # auc_val = AUC(pred_val.detach().numpy(), ytime_val, ystatus_val)

            with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_randomseed' + str(k_num) + '.log',
                      'a') as f:
                f.writelines(
                    'Iteration:' + str(iter) + ",cind:" + str(cind_val) + '\n')
            if cind_train - cind_val > 0.05:
                if cind_val >= max_cind:
                    max_cind = cind_val
                    best_iter = iter
                    

                # auc_val = AUC(pred_val.detach().numpy(), ytime_val, ystatus_val)
            if  iter > 100:
                    break

        max_cind_list.append(max_cind)
        with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_AVE_Cind' + '.log', 'a') as f:
            f.writelines(str(best_iter) + ',' + str(max_cind) + '\n')
    with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_AVE_Cind' + '.log', 'a') as f:
        f.writelines("max cind ave:" + str(np.mean(np.array(max_cind_list))) + '\n')
        f.writelines("max cind std:" + str(np.std(np.array(max_cind_list))) + '\n')
