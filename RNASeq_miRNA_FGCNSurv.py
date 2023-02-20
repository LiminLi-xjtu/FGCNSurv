import pandas as pd
from sklearn.model_selection import KFold
import sys
sys.path.append("..")
import numpy as np
import torch
import random
# import math
# import argparse
# import json
from model.neural_network import Graph_Survival_Analysis
# import torch.utils.data as Data
from sklearn import preprocessing


#
# def AUC(pred, ytime_test, ystatus_test):
#     N_test = ystatus_test.shape[0]
#     ystatus_test = np.squeeze(ystatus_test)
#     ytime_test = np.squeeze(ytime_test)
#     theta = np.squeeze(pred)
#     total = 0
#     count = 0
#
#     for i in range(N_test):
#         if ystatus_test[i] == 1:
#             for j in range(N_test):
#                 # if ytime_test[i] < quantile_2 < ytime_test[j]:
#                 if ytime_test[i] < 365 * 1 < ytime_test[j]:
#                     total = total + 1
#                     if theta[j] < theta[i]:
#                         count = count + 1
#                     elif theta[j] == theta[i]:
#                         count = count + 0.5
#
#     for i in range(N_test):
#         if ystatus_test[i] == 1:
#             for j in range(N_test):
#                 # if ytime_test[i] < quantile_2 < ytime_test[j]:
#                 if ytime_test[i] < 365 * 5 < ytime_test[j]:
#                     total = total + 1
#                     if theta[j] < theta[i]:
#                         count = count + 1
#                     elif theta[j] == theta[i]:
#                         count = count + 0.5
#
#     for i in range(N_test):
#         if ystatus_test[i] == 1:
#             for j in range(N_test):
#                 # if ytime_test[i] < quantile_2 < ytime_test[j]:
#                 if ytime_test[i] < 365 * 10 < ytime_test[j]:
#                     total = total + 1
#                     if theta[j] < theta[i]:
#                         count = count + 1
#                     elif theta[j] == theta[i]:
#                         count = count + 0.5
#
#     return count / total
#

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
                        eav_count  = eav_count +1
                        print("相等的对数")
                        print(eav_count)
    return concord / total


if __name__ == '__main__':

    data_base_path = "/share/home/4120107034/FGCNSurv/Datasets_and_Preprocessing/data/matched_RNASeq_miRNA_clinical/BRCA"
    RNASeq_dataframe = pd.read_csv(data_base_path + "/RNASeq.csv").iloc[:, 0:6000]
    clinical_dataframe = pd.read_csv(data_base_path + "/clinical.csv")
    miRNA_dataframe = pd.read_csv(data_base_path + "/miRNA.csv")

    train_lr = 2e-4
    RNASeq_feature = np.array(RNASeq_dataframe)
    miRNA_feature = np.array(miRNA_dataframe)

    standard_scaler = preprocessing.StandardScaler()
    RNASeq_feature = standard_scaler.fit_transform(RNASeq_feature)
    miRNA_feature = standard_scaler.fit_transform(miRNA_feature)

    standard_scaler = preprocessing.StandardScaler()
    RNASeq_feature = standard_scaler.fit_transform(RNASeq_feature)
    miRNA_feature = standard_scaler.fit_transform(miRNA_feature)

    row_num, col_num = RNASeq_feature.shape
    for j_col in range(col_num):
        percentiles = np.array([10, 30, 50, 70, 90])
        percentiles_val = np.array([5, 20, 40, 60, 80, 95])
        ptiles_vers = np.percentile(RNASeq_feature[:, j_col], percentiles)
        percentiles_val_vers = np.percentile(RNASeq_feature[:, j_col], percentiles_val)
        for i_row in range(row_num):

            if RNASeq_feature[i_row, j_col] < ptiles_vers[0]:
                RNASeq_feature[i_row, j_col] = percentiles_val_vers[0]
            if RNASeq_feature[i_row, j_col] >= ptiles_vers[4]:
                RNASeq_feature[i_row, j_col] = percentiles_val_vers[5]
            if ptiles_vers[0] <= RNASeq_feature[i_row, j_col] < ptiles_vers[1]:
                RNASeq_feature[i_row, j_col] = percentiles_val_vers[1]
            if ptiles_vers[1] <= RNASeq_feature[i_row, j_col] < ptiles_vers[2]:
                RNASeq_feature[i_row, j_col] = percentiles_val_vers[2]
            if ptiles_vers[2] <= RNASeq_feature[i_row, j_col] < ptiles_vers[3]:
                RNASeq_feature[i_row, j_col] = percentiles_val_vers[3]
            if ptiles_vers[3] <= RNASeq_feature[i_row, j_col] < ptiles_vers[4]:
                RNASeq_feature[i_row, j_col] = percentiles_val_vers[4]

    row_num, col_num = miRNA_feature.shape
    for j_col in range(col_num):
        percentiles = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        percentiles_val = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
        ptiles_vers = np.percentile(miRNA_feature[:, j_col], percentiles)
        percentiles_val_vers = np.percentile(miRNA_feature[:, j_col], percentiles_val)
        for i_row in range(row_num):

            if miRNA_feature[i_row, j_col] < ptiles_vers[0]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[0]
            if miRNA_feature[i_row, j_col] >= ptiles_vers[8]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[9]
            if ptiles_vers[0] <= miRNA_feature[i_row, j_col] < ptiles_vers[1]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[1]
            if ptiles_vers[1] <= miRNA_feature[i_row, j_col] < ptiles_vers[2]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[2]
            if ptiles_vers[2] <= miRNA_feature[i_row, j_col] < ptiles_vers[3]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[3]
            if ptiles_vers[3] <= miRNA_feature[i_row, j_col] < ptiles_vers[4]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[4]
            if ptiles_vers[4] <= miRNA_feature[i_row, j_col] < ptiles_vers[5]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[5]
            if ptiles_vers[5] <= miRNA_feature[i_row, j_col] < ptiles_vers[6]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[6]
            if ptiles_vers[6] <= miRNA_feature[i_row, j_col] < ptiles_vers[7]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[7]
            if ptiles_vers[7] <= miRNA_feature[i_row, j_col] < ptiles_vers[8]:
                miRNA_feature[i_row, j_col] = percentiles_val_vers[8]

    clinical_feature = np.array(clinical_dataframe)

    for random_num in [1, 2, 3, 4, 5]:
    # for random_num in [1]:
        kf = KFold(n_splits=5, random_state=random_num, shuffle=True)
        max_cind_list = []
        ystatus_dead_index = np.squeeze(np.argwhere(clinical_feature[:, 0] == 1))
        ystatus_censor_index = np.squeeze(np.argwhere(clinical_feature[:, 0] == 0))

        RNASeq_dead = RNASeq_feature[ystatus_dead_index,]
        RNASeq_censor = RNASeq_feature[ystatus_censor_index,]
        miRNA_dead = miRNA_feature[ystatus_dead_index,]
        miRNA_censor = miRNA_feature[ystatus_censor_index,]
        clinical_dead = clinical_feature[ystatus_dead_index,]
        clinical_censor = clinical_feature[ystatus_censor_index,]

        k_num = 1
        for ind_train_dead, ind_val_dead in kf.split(RNASeq_dead):
            if k_num == 1:
                gene_dead_train_1 = RNASeq_dead[ind_train_dead,]
                miRNA_dead_train_1 = miRNA_dead[ind_train_dead,]
                clinical_dead_train_1 = clinical_dead[ind_train_dead,]

                gene_dead_val_1 = RNASeq_dead[ind_val_dead,]
                miRNA_dead_val_1 = miRNA_dead[ind_val_dead,]
                clinical_dead_val_1 = clinical_dead[ind_val_dead,]
            if k_num == 2:
                gene_dead_train_2 = RNASeq_dead[ind_train_dead,]
                miRNA_dead_train_2 = miRNA_dead[ind_train_dead,]
                clinical_dead_train_2 = clinical_dead[ind_train_dead,]

                gene_dead_val_2 = RNASeq_dead[ind_val_dead,]
                miRNA_dead_val_2 = miRNA_dead[ind_val_dead,]
                clinical_dead_val_2 = clinical_dead[ind_val_dead,]
            if k_num == 3:
                gene_dead_train_3 = RNASeq_dead[ind_train_dead,]
                miRNA_dead_train_3 = miRNA_dead[ind_train_dead,]
                clinical_dead_train_3 = clinical_dead[ind_train_dead,]

                gene_dead_val_3 = RNASeq_dead[ind_val_dead,]
                miRNA_dead_val_3 = miRNA_dead[ind_val_dead,]
                clinical_dead_val_3 = clinical_dead[ind_val_dead,]
            if k_num == 4:
                gene_dead_train_4 = RNASeq_dead[ind_train_dead,]
                miRNA_dead_train_4 = miRNA_dead[ind_train_dead,]
                clinical_dead_train_4 = clinical_dead[ind_train_dead,]

                gene_dead_val_4 = RNASeq_dead[ind_val_dead,]
                miRNA_dead_val_4 = miRNA_dead[ind_val_dead,]
                clinical_dead_val_4 = clinical_dead[ind_val_dead,]
            if k_num == 5:
                gene_dead_train_5 = RNASeq_dead[ind_train_dead,]
                miRNA_dead_train_5 = miRNA_dead[ind_train_dead,]
                clinical_dead_train_5 = clinical_dead[ind_train_dead,]

                gene_dead_val_5 = RNASeq_dead[ind_val_dead,]
                miRNA_dead_val_5 = miRNA_dead[ind_val_dead,]
                clinical_dead_val_5 = clinical_dead[ind_val_dead,]
            k_num = k_num + 1

        k_num = 1
        for ind_train_censor, ind_val_censor in kf.split(RNASeq_censor):
            if k_num == 1:
                gene_censor_train_1 = RNASeq_censor[ind_train_censor,]
                miRNA_censor_train_1 = miRNA_censor[ind_train_censor,]
                clinical_censor_train_1 = clinical_censor[ind_train_censor,]

                gene_censor_val_1 = RNASeq_censor[ind_val_censor,]
                miRNA_censor_val_1 = miRNA_censor[ind_val_censor,]
                clinical_censor_val_1 = clinical_censor[ind_val_censor,]

            if k_num == 2:
                gene_censor_train_2 = RNASeq_censor[ind_train_censor,]
                miRNA_censor_train_2 = miRNA_censor[ind_train_censor,]
                clinical_censor_train_2 = clinical_censor[ind_train_censor,]

                gene_censor_val_2 = RNASeq_censor[ind_val_censor,]
                miRNA_censor_val_2 = miRNA_censor[ind_val_censor,]
                clinical_censor_val_2 = clinical_censor[ind_val_censor,]

            if k_num == 3:
                gene_censor_train_3 = RNASeq_censor[ind_train_censor,]
                miRNA_censor_train_3 = miRNA_censor[ind_train_censor,]
                clinical_censor_train_3 = clinical_censor[ind_train_censor,]

                gene_censor_val_3 = RNASeq_censor[ind_val_censor,]
                miRNA_censor_val_3 = miRNA_censor[ind_val_censor,]
                clinical_censor_val_3 = clinical_censor[ind_val_censor,]

            if k_num == 4:
                gene_censor_train_4 = RNASeq_censor[ind_train_censor,]
                miRNA_censor_train_4 = miRNA_censor[ind_train_censor,]
                clinical_censor_train_4 = clinical_censor[ind_train_censor,]

                gene_censor_val_4 = RNASeq_censor[ind_val_censor,]
                miRNA_censor_val_4 = miRNA_censor[ind_val_censor,]
                clinical_censor_val_4 = clinical_censor[ind_val_censor,]
            if k_num == 5:
                gene_censor_train_5 = RNASeq_censor[ind_train_censor,]
                miRNA_censor_train_5 = miRNA_censor[ind_train_censor,]
                clinical_censor_train_5 = clinical_censor[ind_train_censor,]

                gene_censor_val_5 = RNASeq_censor[ind_val_censor,]
                miRNA_censor_val_5 = miRNA_censor[ind_val_censor,]
                clinical_censor_val_5 = clinical_censor[ind_val_censor,]

            k_num = k_num + 1

        for k_num in range(1, 6):
            if k_num == 1:
                RNASeq_train = np.concatenate((gene_dead_train_1, gene_censor_train_1), axis=0)
                miRNA_train = np.concatenate((miRNA_dead_train_1, miRNA_censor_train_1), axis=0)
                clinical_train = np.squeeze(np.concatenate((clinical_dead_train_1, clinical_censor_train_1), axis=0))

                RNASeq_val = np.concatenate((gene_dead_val_1, gene_censor_val_1), axis=0)
                miRNA_val = np.concatenate((miRNA_dead_val_1, miRNA_censor_val_1), axis=0)
                clinical_val = np.squeeze(np.concatenate((clinical_dead_val_1, clinical_censor_val_1), axis=0))
            if k_num == 2:
                RNASeq_train = np.concatenate((gene_dead_train_2, gene_censor_train_2), axis=0)
                miRNA_train = np.concatenate((miRNA_dead_train_2, miRNA_censor_train_2), axis=0)
                clinical_train = np.squeeze(np.concatenate((clinical_dead_train_2, clinical_censor_train_2), axis=0))

                RNASeq_val = np.concatenate((gene_dead_val_2, gene_censor_val_2), axis=0)
                miRNA_val = np.concatenate((miRNA_dead_val_2, miRNA_censor_val_2), axis=0)
                clinical_val = np.squeeze(np.concatenate((clinical_dead_val_2, clinical_censor_val_2), axis=0))
            if k_num == 3:
                RNASeq_train = np.concatenate((gene_dead_train_3, gene_censor_train_3), axis=0)
                miRNA_train = np.concatenate((miRNA_dead_train_3, miRNA_censor_train_3), axis=0)
                clinical_train = np.squeeze(np.concatenate((clinical_dead_train_3, clinical_censor_train_3), axis=0))

                RNASeq_val = np.concatenate((gene_dead_val_3, gene_censor_val_3), axis=0)
                miRNA_val = np.concatenate((miRNA_dead_val_3, miRNA_censor_val_3), axis=0)
                clinical_val = np.squeeze(np.concatenate((clinical_dead_val_3, clinical_censor_val_3), axis=0))
            if k_num == 4:
                RNASeq_train = np.concatenate((gene_dead_train_4, gene_censor_train_4), axis=0)
                miRNA_train = np.concatenate((miRNA_dead_train_4, miRNA_censor_train_4), axis=0)
                clinical_train = np.squeeze(np.concatenate((clinical_dead_train_4, clinical_censor_train_4), axis=0))

                RNASeq_val = np.concatenate((gene_dead_val_4, gene_censor_val_4), axis=0)
                miRNA_val = np.concatenate((miRNA_dead_val_4, miRNA_censor_val_4), axis=0)
                clinical_val = np.squeeze(np.concatenate((clinical_dead_val_4, clinical_censor_val_4), axis=0))
            if k_num == 5:
                RNASeq_train = np.concatenate((gene_dead_train_5, gene_censor_train_5), axis=0)
                miRNA_train = np.concatenate((miRNA_dead_train_5, miRNA_censor_train_5), axis=0)
                clinical_train = np.squeeze(np.concatenate((clinical_dead_train_5, clinical_censor_train_5), axis=0))

                RNASeq_val = np.concatenate((gene_dead_val_5, gene_censor_val_5), axis=0)
                miRNA_val = np.concatenate((miRNA_dead_val_5, miRNA_censor_val_5), axis=0)
                clinical_val = np.squeeze(np.concatenate((clinical_dead_val_5, clinical_censor_val_5), axis=0))

            torch.manual_seed(1)
            model = Graph_Survival_Analysis()

            RNASeq_val_tensor = torch.tensor(RNASeq_val, dtype=torch.float)
            miRNA_val_tensor = torch.tensor(miRNA_val, dtype=torch.float)
            ytime_val = np.squeeze(clinical_val[:, 1])
            ystatus_val = np.squeeze(clinical_val[:, 0])
            model.eval()
            pred_val = model.get_survival_result(RNASeq_val_tensor, miRNA_val_tensor)
            cind_val = CIndex(pred_val.detach().numpy(), ytime_val, ystatus_val)
            with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_randomseed' + str(k_num) + '.log', 'a') as f:
                f.writelines('Iteration:' + str(0) + ",cind:" + str(cind_val) + '\n')
            max_cind = 0
            count = 0
            optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=train_lr, weight_decay=5e-4)
            iter = 0
            while True:
                iter = iter + 1
                index = np.squeeze(np.arange(0, RNASeq_train.shape[0]))
                fir_index = np.random.choice(index, size=400, replace=False)
                sen_index = set(index.tolist()) - set(fir_index)

                fir_index = np.array(fir_index)
                sen_index = np.array(list(sen_index))
                ystatus_train = np.squeeze(clinical_train[:, 0])
                ytime_train = np.squeeze(clinical_train[:, 1])

                RNASeq_batch_train = RNASeq_train[fir_index,]
                RNASeq_train_tensor = torch.tensor(RNASeq_batch_train, dtype=torch.float)
                miRNA_batch_train = miRNA_train[fir_index,]
                miRNA_train_tensor = torch.tensor(miRNA_batch_train, dtype=torch.float)

                ystatus_batch_train = ystatus_train[fir_index,]
                ystatus_train_tensor = torch.tensor(ystatus_batch_train, dtype=torch.float)

                ytime_batch_train = ytime_train[fir_index,]
                ytime_train_tensor = torch.tensor(ytime_batch_train, dtype=torch.float)

                real_batch_size = RNASeq_train_tensor.shape[0]
                R_matrix_batch_train = torch.tensor(np.zeros([real_batch_size, real_batch_size], dtype=int),
                                                    dtype=torch.float)

                for i in range(real_batch_size):
                    R_matrix_batch_train[i,] = torch.tensor(
                        np.array(list(map(int, (ytime_train_tensor >= ytime_train_tensor[i])))))
                model.train()
                theta = model.get_survival_result(RNASeq_train_tensor, miRNA_train_tensor)
                exp_theta = torch.reshape(torch.exp(theta), [real_batch_size])
                theta = torch.reshape(theta, [real_batch_size])
                fuse_loss = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta,
                                                                                         R_matrix_batch_train),
                                                                               dim=1))),
                                                  torch.reshape(ystatus_train_tensor, [real_batch_size])))

                print('cox损失:' + str(fuse_loss))
                print("迭代次数：" + str(iter))
                loss = fuse_loss
                # Update meta-parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # end = time.time()
                # print("1 iteration time:", end - start)

                RNASeq_batch_train = RNASeq_train[sen_index,]
                RNASeq_train_tensor = torch.tensor(RNASeq_batch_train, dtype=torch.float)
                miRNA_batch_train = miRNA_train[sen_index,]
                miRNA_train_tensor = torch.tensor(miRNA_batch_train, dtype=torch.float)

                ystatus_batch_train = ystatus_train[sen_index,]
                ystatus_train_tensor = torch.tensor(ystatus_batch_train, dtype=torch.float)

                ytime_batch_train = ytime_train[sen_index,]
                ytime_train_tensor = torch.tensor(ytime_batch_train, dtype=torch.float)

                real_batch_size = RNASeq_train_tensor.shape[0]
                R_matrix_batch_train = torch.tensor(np.zeros([real_batch_size, real_batch_size], dtype=int),
                                                    dtype=torch.float)

                for i in range(real_batch_size):
                    R_matrix_batch_train[i,] = torch.tensor(
                        np.array(list(map(int, (ytime_train_tensor >= ytime_train_tensor[i])))))
                model.train()
                theta = model.get_survival_result(RNASeq_train_tensor, miRNA_train_tensor)
                exp_theta = torch.reshape(torch.exp(theta), [real_batch_size])
                theta = torch.reshape(theta, [real_batch_size])
                fuse_loss = -torch.mean(torch.mul((theta - torch.log(torch.sum(torch.mul(exp_theta,
                                                                                         R_matrix_batch_train),
                                                                               dim=1))),
                                                  torch.reshape(ystatus_train_tensor, [real_batch_size])))

                print('cox损失:' + str(fuse_loss))
                print("迭代次数：" + str(iter))
                loss = fuse_loss
                # Update meta-parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # end = time.time()
                # print("1 iteration time:", end - start)
                model.eval()
                RNASeq_train_tensor = torch.tensor(RNASeq_train, dtype=torch.float)
                miRNA_train_tensor = torch.tensor(miRNA_train, dtype=torch.float)
                pred_train = model.get_survival_result(RNASeq_train_tensor, miRNA_train_tensor)
                cind_train = CIndex(pred_train, ytime_train, ystatus_train)
                # auc_train = AUC(pred_train, ytime_train, ystatus_train)

                with open('Cox_Result_lr5e-05/Cox_Train/Regularizer_eval_randomseed' + str(k_num) + '.log',
                          'a') as f:
                    f.writelines('Iteration:' + str(iter) + ",cind_train:" + str(cind_train) + '\n')

                RNASeq_val_tensor = torch.tensor(RNASeq_val, dtype=torch.float)
                miRNA_val_tensor = torch.tensor(miRNA_val, dtype=torch.float)

                pred_val = model.get_survival_result(RNASeq_val_tensor, miRNA_val_tensor)
                cind_val = CIndex(pred_val.detach().numpy(), ytime_val, ystatus_val)
                # auc_val = AUC(pred_val.detach().numpy(), ytime_val, ystatus_val)

                with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_randomseed' + str(k_num) + '.log',
                          'a') as f:
                    f.writelines(
                        'Iteration:' + str(iter) + ",cind:" + str(cind_val) + '\n')

                if cind_val >= max_cind and iter > 5:
                    max_cind = cind_val
                    count = iter
                if iter > 100:
                    break
            max_cind_list.append(max_cind)

            with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_AVE_Cind' + '.log', 'a') as f:
                f.writelines(str(count) + ',' + str(max_cind) + '\n')
        with open('Cox_Result_lr5e-05/Cox_Val/Regularizer_eval_AVE_Cind' + '.log', 'a') as f:
            f.writelines("max cind ave:" + str(np.mean(np.array(max_cind_list))) + '\n')
            f.writelines("max cind std:" + str(np.std(np.array(max_cind_list))) + '\n')