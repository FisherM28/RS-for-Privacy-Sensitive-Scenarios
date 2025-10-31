import copy
import math
import pickle
import random
import sys
from statistics import mean
import numpy as np
import torch
import torch.optim as optim
from models.localGCN import GCN
from utility.helper import *
from utility.batch_test import *
import warnings

warnings.filterwarnings('ignore')
from time import time
from datetime import datetime 

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


if __name__ == '__main__':
    args.device = (
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    )
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    args.mode = 'fed'
    # args.include_feature = False

    current_time = datetime.now().strftime("%m%d_%H%M%S")
    log_file_name = f"log/FedAvg_{args.model_type}_{args.dataset}_{current_time}.txt"
    tee = Tee(log_file_name, "w")
    model = GCN(data_generator.n_users, data_generator.n_items, args).to(args.device)
    
    print(args)
    print(f'\nStart Training')
    
    t0 = time()
    n_fed_client_each_round = args.clients
    n_client = data_generator.n_users

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_hr_5, best_hr_10, best_hr_20, best_ndcg_5, best_ndcg_10, best_ndcg_20 = 0, 0, 0, 0, 0, 0
    early_stopping_counter = 0
    early_stopping_limit = args.patience

    training_time = (0.0,)
    begin_time = time()
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
        n_batch = data_generator.n_train // args.batch_size + 1

        model_para_list = []
        user_ini_state = copy.deepcopy(model.state_dict())['embedding_dict.user_emb']
        user_emb_list = {}
        idxs_users = random.sample(range(0, n_client), n_fed_client_each_round)

        for idx in idxs_users:
            model_ini = copy.deepcopy(model.state_dict())
            users, pos_items, neg_items = data_generator.sample(idx)
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(
                users, pos_items, neg_items, drop_flag=args.node_dropout_flag
            )

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
            )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

            model_aft = copy.deepcopy(model.state_dict())

            user_emb_list[idx] = model_aft['embedding_dict.user_emb'][idx]

            model_para_list += [model_aft]
            model.load_state_dict(model_ini)

        w_ = FedAvg(model_para_list)
        # update user embedding
        for j in user_emb_list:
            user_ini_state[j] = user_emb_list[j]
        w_['embedding_dict.user_emb'] = copy.deepcopy(user_ini_state)
        model.load_state_dict(w_)

        # negative testing
        hr_list_5 = []
        ndcg_list_5 = []
        hr_list_10 = []
        ndcg_list_10 = []
        hr_list_20 = []
        ndcg_list_20 = []
        for test_idx in range(data_generator.n_users):
            test_positive, test_nagetive = data_generator.sample_test_nagative(test_idx)
            u_g_embeddings, pos_i_g_embeddings = model.get_u_i_embedding([test_idx], test_positive)
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()[0]

            u_g_embeddings, pos_i_g_embeddings = model.get_u_i_embedding([test_idx] * len(test_nagetive),
                                                                         test_nagetive)
            rate_batch_nagetive = torch.matmul(u_g_embeddings.unsqueeze(1),
                                               pos_i_g_embeddings.unsqueeze(2)).squeeze().detach().cpu()

            torch_cat = torch.cat((rate_batch, rate_batch_nagetive), 0).numpy()
            ranking = list(np.argsort(torch_cat))[::-1].index(0) + 1

            ndcg = 0
            hr = 0
            if ranking <= 5:
                hr = 1
                ndcg = math.log(2) / math.log(1 + ranking)
            hr_list_5.append(hr), ndcg_list_5.append(ndcg)

            ndcg = 0
            hr = 0
            if ranking <= 10:
                hr = 1
                ndcg = math.log(2) / math.log(1 + ranking)
            hr_list_10.append(hr), ndcg_list_10.append(ndcg)

            ndcg = 0
            hr = 0
            if ranking <= 20:
                hr = 1
                ndcg = math.log(2) / math.log(1 + ranking)
            hr_list_20.append(hr), ndcg_list_20.append(ndcg)

        print(f'Epoch {epoch} [{time() - t1:.1f}s]: loss = {loss:.4f}, HR@5 = {mean(hr_list_5):.4f}, NDCG@5 = {mean(ndcg_list_5):.4f}, HR@10 = {mean(hr_list_10):.4f}, NDCG@10 = {mean(ndcg_list_10):.4f}, HR@20 = {mean(hr_list_20):.4f}, NDCG@20 = {mean(ndcg_list_20):.4f}')
        if mean(hr_list_5) > best_hr_5 or mean(ndcg_list_5) > best_ndcg_5 or mean(hr_list_10) > best_hr_10 or mean(ndcg_list_10) > best_ndcg_10 or mean(hr_list_20) > best_hr_20 or mean(ndcg_list_20) > best_ndcg_20:
            early_stopping_counter = 0
            best_hr_5 = max(best_hr_5, mean(hr_list_5))
            best_ndcg_5 = max(best_ndcg_5, mean(ndcg_list_5))
            best_hr_10 = max(best_hr_10, mean(hr_list_10))
            best_ndcg_10 = max(best_ndcg_10, mean(ndcg_list_10))
            best_hr_20 = max(best_hr_20, mean(hr_list_20))
            best_ndcg_20 = max(best_ndcg_20, mean(ndcg_list_20))
            training_time = time() - begin_time
        else:
            early_stopping_counter += 1
            training_time = time() - begin_time

        if early_stopping_counter >= early_stopping_limit:
            print(f'\nEarly stopping after {epoch} epochs.')
            print(f"Best Result: HR@5 = {'%.4f' % best_hr_5}, NDCG@5 = {'%.4f' % best_ndcg_5}, HR@10 = {'%.4f' % best_hr_10}, NDCG@10 = {'%.4f' % best_ndcg_10}, HR@20 = {'%.4f' % best_hr_20}, NDCG@20 = {'%.4f' % best_ndcg_20}")
            print(f'Trainig time: {training_time:.1f}s\n')
            break

        if (epoch + 1) % 5 == 0:
            print(f"\nBest Result: HR@5 = {'%.4f' % best_hr_5}, NDCG@5 = {'%.4f' % best_ndcg_5}, HR@10 = {'%.4f' % best_hr_10}, NDCG@10 = {'%.4f' % best_ndcg_10}, HR@20 = {'%.4f' % best_hr_20}, NDCG@20 = {'%.4f' % best_ndcg_20}")
            print(f'Trainig time: {training_time:.1f}s\n')

    if args.save_flag == 1:
        current_time = datetime.now().strftime("%m%d_%H%M%")
        model_path = 'saved_model/' + 'FedAvg_' + args.model_type + '_' + current_time + '.pkl'
        torch.save(model.state_dict(), model_path)
        print('Save the weights in path:', model_path)