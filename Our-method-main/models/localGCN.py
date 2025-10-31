import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args):
        super(GCN, self).__init__()
        self.n_users = n_user
        self.n_items = n_item
        self.n_fold = 1
        # load parameters info
        self.device = args.device
        self.batch_size = args.batch_size
        self.groups = args.groups
        self.emb_size = args.embed_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = args.layers
        self.decay = eval(args.regs)[0]
        self.node_dropout = args.node_dropout[0]

        self.norm_adj = norm_adj

        self.eps = args.eps  # Epsilon for adversarial weights
        self.reg_adv = args.reg_adv  # Regularization for adversarial loss

        # init parameters
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict(
            {
                'user_emb': nn.Parameter(
                    initializer(torch.empty(self.n_users, self.emb_size))
                ),
                'item_emb': nn.Parameter(
                    initializer(torch.empty(self.n_items, self.emb_size))
                ),
            }
        )

        self.W_gc_1 = nn.Parameter(
            initializer(torch.empty(self.emb_size, self.emb_size))
        )
        self.b_gc_1 = nn.Parameter(initializer(torch.empty(1, self.emb_size)))
        self.W_gc_2 = nn.Parameter(
            initializer(torch.empty(self.emb_size, self.emb_size))
        )
        self.b_gc_2 = nn.Parameter(initializer(torch.empty(1, self.emb_size)))
        self.W_gc = nn.Parameter(initializer(torch.empty(self.emb_size, self.groups)))
        self.b_gc = nn.Parameter(initializer(torch.empty(1, self.groups)))

        self.A_fold_hat = self._split_A_hat(self.norm_adj)
        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

    def pre_epoch_processing(self):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col])  # .transpose()
        indices = torch.from_numpy(indices).type(torch.LongTensor)
        data = torch.from_numpy(coo.data)
        return torch.sparse.FloatTensor(
            indices, data, torch.Size((coo.shape[0], coo.shape[1]))
        )

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(
                self._convert_sp_mat_to_sp_tensor(X[start:end]).to(self.device)
            )
        return A_fold_hat

    def sparse_dense_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        ret_tensor = torch.sparse.FloatTensor(i, v * dv, s.size())
        return ret_tensor.to(self.device)

    def _split_A_hat_group(self, X, group_embedding):
        group_embedding = group_embedding.T
        A_fold_hat_group = []
        A_fold_hat_group_filter = []
        A_fold_hat = self.A_fold_hat

        # split L in fold
        fold_len = (self.n_users + self.n_items) // self.n_fold

        # k groups
        for k in range(0, self.groups):
            A_fold_item_filter = []
            A_fold_hat_item = []

            # n folds in per group (filter user)
            for i_fold in range(self.n_fold):
                start = i_fold * fold_len
                if i_fold == self.n_fold - 1:
                    end = self.n_users + self.n_items
                else:
                    end = (i_fold + 1) * fold_len

                temp_g = self.sparse_dense_mul(
                    A_fold_hat[i_fold],
                    group_embedding[k].expand(A_fold_hat[i_fold].shape),
                )
                temp_slice = self.sparse_dense_mul(
                    temp_g,
                    torch.unsqueeze(group_embedding[k][start:end], dim=1).expand(
                        temp_g.shape
                    ),
                )
                # A_fold_hat_item.append(A_fold_hat[i_fold].__mul__(group_embedding[k]).__mul__(torch.unsqueeze(group_embedding[k][start:end], dim=1)))
                A_fold_hat_item.append(temp_slice)
                item_filter = torch.sparse.sum(
                    A_fold_hat_item[i_fold], dim=1
                ).to_dense()
                item_filter = torch.where(
                    item_filter > 0.0,
                    torch.ones_like(item_filter),
                    torch.zeros_like(item_filter),
                )
                A_fold_item_filter.append(item_filter)

            A_fold_item = torch.concat(A_fold_item_filter, dim=0)
            A_fold_hat_group_filter.append(A_fold_item)
            A_fold_hat_group.append(A_fold_hat_item)

        return A_fold_hat_group, A_fold_hat_group_filter

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        ego_embeddings = torch.cat(
            [self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0
        )
        return ego_embeddings

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def get_u_i_embedding(self, users, pos_items):
        return (
            self.embedding_dict['user_emb'][users],
            self.embedding_dict['item_emb'][pos_items],
        )

    def forward(self, users, pos_items, neg_items, drop_flag=True):
        A_fold_hat = self.A_fold_hat
        ego_embeddings = self.get_ego_embeddings()
        # group users
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))
        user_group_embeddings_side = torch.concat(temp_embed, dim=0) + ego_embeddings

        user_group_embeddings_hidden_1 = F.leaky_relu(
            torch.matmul(user_group_embeddings_side, self.W_gc_1) + self.b_gc_1
        )
        if drop_flag==True:
            user_group_embeddings_hidden_d1 = F.dropout(user_group_embeddings_hidden_1, self.node_dropout)

        user_group_embeddings_sum = (
            torch.matmul(user_group_embeddings_hidden_d1, self.W_gc) + self.b_gc
        )
        # user 0-1
        a_top, a_top_idx = torch.topk(user_group_embeddings_sum, 1, sorted=False)
        user_group_embeddings = torch.eq(user_group_embeddings_sum, a_top).type(
            torch.float32
        )
        u_group_embeddings, i_group_embeddings = torch.split(
            user_group_embeddings, [self.n_users, self.n_items], 0
        )
        i_group_embeddings = torch.ones_like(i_group_embeddings)
        user_group_embeddings = torch.concat(
            [u_group_embeddings, i_group_embeddings], dim=0
        )
        # Matrix mask
        A_fold_hat_group, A_fold_hat_group_filter = self._split_A_hat_group(
            self.norm_adj, user_group_embeddings
        )
        # embedding transformation
        ego_embeddings_0 = ego_embeddings
        all_embeddings = [ego_embeddings]
        temp_embed = []

        for f in range(self.n_fold):
            temp_embed.append(torch.sparse.mm(A_fold_hat[f], ego_embeddings))

        side_embeddings = torch.concat(temp_embed, dim=0)
        all_embeddings += [side_embeddings]

        ego_embeddings_g = []
        ego_embeddings_g_0 = []
        for g in range(0, self.groups):
            ego_embeddings_g.append(ego_embeddings)
            ego_embeddings_g_0.append(ego_embeddings)

        ego_embeddings_f = []
        for k in range(1, self.n_layers):
            for g in range(0, self.groups):
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(torch.sparse.mm(A_fold_hat_group[g][f], ego_embeddings_g[g]))
                side_embeddings = torch.concat(temp_embed, dim=0)
                ego_embeddings_g[g] = ego_embeddings_g[g] + side_embeddings

                # layer refinement mechanism
                _weights = F.cosine_similarity(ego_embeddings_g[g], ego_embeddings_g_0[g], dim=-1)
                ego_embeddings_g[g] = torch.einsum('a,ab->ab', _weights, ego_embeddings_g[g])

                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(torch.sparse.mm(A_fold_hat[f], side_embeddings))
                if k == 1:
                    ego_embeddings_f.append(torch.concat(temp_embed, dim=0))
                else:
                    ego_embeddings_f[g] = torch.concat(temp_embed, dim=0)
            ego_embeddings = torch.sum(torch.stack(ego_embeddings_f, dim=0), dim=0)

            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        users.retain_grad()
        pos_items.retain_grad()
        neg_items.retain_grad()

        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        bpr_loss = -1 * torch.mean(maxi)

        # Backward to get grads
        bpr_loss.backward(retain_graph=True)
        grad_user = users.grad
        grad_pos = pos_items.grad
        grad_neg = neg_items.grad

        # Construct adversarial perturbation
        delta_u = nn.functional.normalize(grad_user, p=2, dim=1) * self.eps
        delta_i = nn.functional.normalize(grad_pos, p=2, dim=1) * self.eps
        delta_j = nn.functional.normalize(grad_neg, p=2, dim=1) * self.eps

        # Add adversarial perturbation to embeddings
        adv_pos_scores = torch.sum(
            torch.mul(users + delta_u, pos_items + delta_i), axis=1
        )
        adv_neg_scores = torch.sum(
            torch.mul(users + delta_u, neg_items + delta_j), axis=1
        )
        adv_maxi = nn.LogSigmoid()(adv_pos_scores - adv_neg_scores)
        apr_loss = -1 * torch.mean(adv_maxi)
        apr_loss = self.reg_adv * apr_loss

        # cul regularizer
        regularizer = (
            torch.norm(users) ** 2
            + torch.norm(pos_items) ** 2
            + torch.norm(neg_items) ** 2
        ) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        total_loss = apr_loss + emb_loss + bpr_loss

        return total_loss, bpr_loss, emb_loss
