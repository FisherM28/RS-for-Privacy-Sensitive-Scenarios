import copy
import random
import math
from statistics import mean
import numpy as np
import torch
import torch.optim as optim
from numpy import array
from models.localGCN import GCN
from utility.helper import *
from utility.batch_test import *
import warnings
# warnings.filterwarnings('ignore')
from time import time
from datetime import datetime 
from sklearn.cluster import KMeans

def FedWeiAvg(w,w_):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w_avg[k], w_[0])
        for i in range(1, len(w)):
            w_avg[k] += (w[i][k] * w_[i])
    return w_avg

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

if __name__ == '__main__':

    n_fed_client_each_round = args.clients
    n_cluster = args.clusters

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    args.mode = 'fed'
    n_client = data_generator.n_users

    current_time = datetime.now().strftime("%m%d_%H%M%S")
    log_file_name = f"log/PerFed_{args.model_type}_{args.dataset}_{current_time}.txt"
    tee = Tee(log_file_name, "w")
    model = GCN(data_generator.n_users, data_generator.n_items, args).to(args.device)

    print(args)

    print(f'\nStart Training')

    t0 = time()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    idxs_users = random.sample(range(0, n_client), n_fed_client_each_round)
    wei_usrs = [1./n_fed_client_each_round] * n_fed_client_each_round

    best_hr_5, best_hr_10, best_hr_20, best_ndcg_5, best_ndcg_10, best_ndcg_20 = 0, 0, 0, 0, 0, 0
    early_stopping_counter = 0
    early_stopping_limit = args.patience

    training_time = 0.0,
    begin_time = time()

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        model_para_list = []
        user_ini_state = copy.deepcopy(model.state_dict())['embedding_dict.user_emb']
        user_emb_list = {}
        local_test = []
        w_cluster =  {i:[] for i in range(n_cluster)}
        local_model_test_res = {}
        def LocalTestNeg(data_generator,test_idx,model):
            test_positive, test_nagetive = data_generator.sample_test_nagative(test_idx)
            u_g_embeddings, pos_i_g_embeddings = model.get_u_i_embedding([test_idx], test_positive)
            rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()[0]
            u_g_embeddings, pos_i_g_embeddings = model.get_u_i_embedding([test_idx] * len(test_nagetive),
                                                                         test_nagetive)
            rate_batch_nagetive = torch.matmul(u_g_embeddings.unsqueeze(1),
                                               pos_i_g_embeddings.unsqueeze(2)).squeeze().detach().cpu()
            torch_cat = torch.cat((rate_batch, rate_batch_nagetive), 0).numpy()
            return torch_cat

        for idx in (idxs_users):
            if epoch>1:
                model.load_state_dict(copy.deepcopy(w_cluster_avg[clu_result[idx]]))
                user_ini_state = copy.deepcopy(model.state_dict())['embedding_dict.user_emb']

            model_ini = copy.deepcopy(model.state_dict())
            users, pos_items, neg_items = data_generator.sample(idx)
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            local_model_test_res[idx] = LocalTestNeg(data_generator,idx,model)
            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            model_aft = copy.deepcopy(model.state_dict())
            for _ in range(100):
                ran_idx = random.randint(0, data_generator.n_items-1)
                loc, scale = 0., 0.02
                s = torch.Tensor(np.random.laplace(loc, scale, args.embed_size)).to('cuda')
                model_aft['embedding_dict.item_emb'][ran_idx] = model_aft['embedding_dict.item_emb'][ran_idx] + s
            model_para_list += [model_aft]

            user_emb_list[idx] = model_aft['embedding_dict.user_emb'][idx]

            if epoch >= 1:
                w_cluster[clu_result[idx] ].append(model_aft)

            model.load_state_dict(model_ini)

        w_ = FedAvg(model_para_list)

        for j in user_emb_list:
            user_ini_state[j] = user_emb_list[j]

        w_['embedding_dict.user_emb'] = copy.deepcopy(user_ini_state)

        if epoch>=1 :
            w_cluster_avg = []
            for i in range(len(w_cluster)):
                try:
                    w_cluster_avg.append(FedAvg(w_cluster[i]))
                except Exception:
                    w_cluster_avg.append(w_)

        # 开始
        res_list = []
        if epoch > 1:
            hr_list_5 = []
            ndcg_list_5 = []
            hr_list_10 = []
            ndcg_list_10 = []
            hr_list_20 = []
            ndcg_list_20 = []
            for i in range(len(w_cluster)):
                users_to_test = [x for x in range(n_client) if clu_result[x]==i]

                for test_idx in (users_to_test):
                    model.load_state_dict(w_cluster_avg[i])
                    torch_cat = LocalTestNeg(data_generator, test_idx, model)

                    model.load_state_dict(w_)
                    torch_cat2 = LocalTestNeg(data_generator, test_idx, model)

                    if local_model_test_res.get(test_idx) is not None:
                        torch_cat = np.mean( np.array([ torch_cat2,torch_cat, local_model_test_res[test_idx] ]), axis=0 )
                    else:
                        torch_cat = np.mean(np.array([torch_cat2,torch_cat]), axis=0)

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
    
        model.load_state_dict(w_)
        users_emb = w_['embedding_dict.user_emb']
        users_emb = users_emb.cpu().detach().numpy()

        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(users_emb)
        clu_result = list(kmeans.labels_)
        clu_result_dict = dict((x, clu_result.count(x)) for x in set(clu_result))
        clu_result_weight = [clu_result_dict[i] for i in clu_result]

        n_rdm = int(n_fed_client_each_round/2)
        n_choice = int(n_fed_client_each_round/2)
        idxs_users_rdm = random.sample(range(0, n_client), n_rdm)
        idxs_users_choice = random.choices(range(0, n_client), weights=(clu_result_weight), k=n_choice)
        idxs_users_ = idxs_users_rdm + idxs_users_choice
        idxs_users_ = list(set(idxs_users_))
        while len(idxs_users_) < n_fed_client_each_round:
            r_= random.sample(range(0, n_client), 1)[0]
            if not r_ in idxs_users_:
                idxs_users_.append(r_)

        idxs_users = idxs_users_
        random.shuffle(idxs_users)

        wei_usrs = [clu_result_weight[i] for i in idxs_users]
        wei_usrs = [i/sum(wei_usrs) for i in wei_usrs]

        if early_stopping_counter >= early_stopping_limit:
            print(f'\nEarly stopping after {epoch} epochs.')
            print(f"Best Result: HR@5 = {'%.4f' % best_hr_5}, NDCG@5 = {'%.4f' % best_ndcg_5}, HR@10 = {'%.4f' % best_hr_10}, NDCG@10 = {'%.4f' % best_ndcg_10}, HR@20 = {'%.4f' % best_hr_20}, NDCG@20 = {'%.4f' % best_ndcg_20}")
            print(f'Trainig time: {training_time:.1f}s\n')
            break

        if (epoch + 1) % 5 == 0:
            print(f"\nBest Result: HR@5 = {'%.4f' % best_hr_5}, NDCG@5 = {'%.4f' % best_ndcg_5}, HR@10 = {'%.4f' % best_hr_10}, NDCG@10 = {'%.4f' % best_ndcg_10}, HR@20 = {'%.4f' % best_hr_20}, NDCG@20 = {'%.4f' % best_ndcg_20}")
            print(f'Trainig time: {training_time:.1f}s\n')

    if args.save_flag == 1:
        current_time = datetime.now().strftime("%m%d_%H%M")
        model_path = 'saved_model/' + 'PerFedRec_' + args.model_type + '_' + current_time + '.pkl'
        torch.save(model.state_dict(), model_path)
        print('Save the weights in path:', model_path)
