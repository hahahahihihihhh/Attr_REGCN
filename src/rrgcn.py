import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc,
                                  rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels,
                 h_dim, opn,
                 num_ent, num_rel, num_attr,
                 sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.num_attr = num_attr
        self.opn = opn
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        # 气象属性编码中的查询向量
        self.q = torch.nn.Parameter(torch.Tensor(1, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.q)
        # 气象属性投影矩阵
        self.W_m = torch.nn.Parameter(torch.Tensor(h_dim, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.W_m)
        # 气象水平投影矩阵
        self.W_vm = torch.nn.Parameter(torch.Tensor(h_dim, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.W_vm)
        # 属性门控参数矩阵
        self.W_g = torch.nn.Parameter(torch.Tensor(2 * h_dim, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.W_g)
        # 属性门控偏置项
        self.b_g = torch.nn.Parameter(torch.Tensor(1, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.b_g)
        # 门控参数矩阵
        self.W_4 = torch.nn.Parameter(torch.Tensor(h_dim, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.W_4)
        # 偏置项
        self.b = torch.nn.Parameter(torch.Tensor(1, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.b)

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError


    def forward(self, g_list, use_cuda):  # attr-regcn: t-k+1,...,t -> t+1
        gate_list = []
        degree_list = []

        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        history_embs = []
        for i, g in enumerate(g_list):
            print(f"{i}/{len(g_list)}")
            rel_g, attr_g = g
            rel_g = rel_g.to(self.gpu) if use_cuda else rel_g
            attr_g = attr_g.to(self.gpu) if use_cuda else attr_g
            pre_V = self.h[:self.num_ent]
            # 气象属性编码单元
            src, dst = attr_g.edges()
            alpha_vm = torch.zeros(self.num_ent, self.num_attr).to(self.gpu)
            rel = attr_g.edata['type']
            assert len(src) == self.num_ent * self.num_attr
            for _ in range(self.num_ent * self.num_attr):
                v, m, l = src[_], rel[_] - self.num_rel, dst[_]
                emb_v, emb_m, emb_l = self.h[v], self.emb_rel[m], self.h[l]
                alpha_vm[v, m] = (self.q @ torch.tanh(self.W_m @ emb_m + self.W_vm @ emb_l)).squeeze()
            alpha_vm = torch.softmax(alpha_vm, dim=1)
            M = torch.zeros(self.num_ent, self.h_dim).to(self.gpu)
            for _ in range(self.num_ent * self.num_attr):
                v, m, l = src[_], rel[_] - self.num_rel, dst[_]
                emb_l = self.h[l]
                M[v] += alpha_vm[v, m] * emb_l
            G = torch.sigmoid((torch.concat([pre_V, M], dim=1)) @ self.W_g + self.b_g)
            V_attr = (1 - G) * pre_V + G * M
            V_attr = V_attr[:self.num_ent]
            # 结构感知的状态传递单元
            V_P = self.rgcn.forward(rel_g, pre_V, [self.emb_rel, self.emb_rel])  # h(t-1) & r(t) -> hw(t)
            V_P = F.normalize(V_P) if self.layer_norm else V_P
            # 门控融合单元
            U = F.sigmoid(V_attr @ self.W_4 + self.b)
            V_met = U * V_P + (1 - U) * V_attr
            tmp_h = self.h.clone()
            tmp_h[:self.num_ent] = V_met
            self.h = tmp_h
            history_embs.append(self.h)
        return history_embs, self.emb_rel, gate_list, degree_list

    def predict(self, test_graph, num_rels, test_triplets, use_cuda):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, r_emb, _, _ = self.forward(test_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel

    def get_loss(self, glist, triples, use_cuda):
        """
        :param glist:
        :param triplets:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu) if use_cuda else all_triples  # all_triples = all_triples.to(self.gpu)

        evolve_embs, r_emb, _, _ = self.forward(glist, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
     
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        return loss_ent, loss_rel
