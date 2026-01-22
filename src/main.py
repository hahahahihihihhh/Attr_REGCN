# @Time    : 2019-08-10 11:20
# @Author  : Lee_zix
# @Email   : Lee_zix@163.com
# @File    : main.py
# @Software: PyCharm
"""
The entry of the KGEvolve
"""

import argparse
import itertools
import json
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
import random

sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph_rel_attr, sanitize_filename
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
import torch.nn.functional as F


dataset = "NYCTAXI20140103"  # 你可改成自己的 dataset
with open("settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
cfg = settings[dataset]


# -------------------------- 日志配置 --------------------------
def setup_logger():
    """配置日志：同时输出到控制台和文件，文件按大小轮转"""
    # 日志格式
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 全局日志级别

    # 1. 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    # 2. 文件处理器（轮转，避免单个文件过大）
    log_dir = f"../logs/{dataset}"
    os.makedirs(log_dir, exist_ok=True)
    if cfg['evolve']:
        flag = 'EVOLVE'
    elif cfg['test']:
        flag = 'TEST'
    else:
        flag = 'TRAIN'
    log_file = os.path.join(log_dir, f"{flag}-d{cfg['n_hidden']}_l{cfg['n_layers']}")
    file_handler = logging.FileHandler(
        log_file,
        mode='w',  # 覆盖写入
        encoding='utf-8'
    )
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)

    # 添加处理器（避免重复添加）
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# 初始化日志
logger = setup_logger()
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
def evolve(model, evolve_list, num_ent, num_nodes, use_cuda, model_name, embs_path):
    # evolve mode: load parameter form file
    if use_cuda:
        checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
    else:
        checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
    logger.info(f"Load Model name: {model_name}. Using best epoch : {checkpoint['epoch']}")  # use best stat checkpoint
    logger.info("\n" + "-" * 10 + "start evolving" + "-" * 10 + "\n")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # evolve mode: start evolve
    g_list = [build_sub_graph_rel_attr(num_nodes, num_ent, g, use_cuda, args.gpu) for g in evolve_list]
    device = model.gpu if use_cuda else torch.device("cpu")
    # 用局部变量 h（不要写 self.h）
    h = F.normalize(model.dynamic_emb, dim=1) if model.layer_norm else model.dynamic_emb
    g_len = len(g_list)
    history_embs = []
    for i, g in enumerate(g_list):
        logger.info(f"evolving -> {i}/{g_len}")
        rel_g, attr_g = g
        rel_g = rel_g.to(device)
        attr_g = attr_g.to(device)
        pre_V = h[:model.num_ent]  # (num_ent, h_dim)
        # ===== 你的向量化属性编码里：把 self.h[...] 改为 h[...] =====
        src, dst = attr_g.edges()
        rel = attr_g.edata["type"]
        m = rel - model.num_rel
        emb_m_e = model.emb_rel[m]
        emb_l_e = h[dst]
        x = (emb_m_e @ model.W_m.T) + (emb_l_e @ model.W_vm.T)
        x = torch.tanh(x)
        score_e = (x * model.q).sum(dim=1)  # q:(1,h_dim) 广播
        alpha_vm = torch.empty(model.num_ent, model.num_attr, device=device, dtype=score_e.dtype)
        alpha_vm.index_put_((src, m), score_e, accumulate=False)
        alpha_vm = torch.softmax(alpha_vm, dim=1)
        L = torch.empty(model.num_ent, model.num_attr, model.h_dim, device=device, dtype=emb_l_e.dtype)
        L.index_put_((src, m), emb_l_e, accumulate=False)
        M = (alpha_vm.unsqueeze(-1) * L).sum(dim=1)
        G = torch.sigmoid(torch.cat([pre_V, M], dim=1) @ model.W_g + model.b_g)
        V_attr = (1.0 - G) * pre_V + G * M
        # ===== 结构传递 =====
        V_P = model.rgcn.forward(rel_g, pre_V, [model.emb_rel, model.emb_rel])
        V_P = F.normalize(V_P, dim=1) if model.layer_norm else V_P
        # ===== 门控融合 =====
        U = torch.sigmoid(V_attr @ model.W_4 + model.b)
        V_met = U * V_P + (1.0 - U) * V_attr
        h = torch.cat([V_met, h[model.num_ent:]], dim=0)
        history_embs.append(h)
    V_mets = torch.stack(history_embs, dim=0)
    V_mets_np = V_mets[:, :model.num_ent, :].detach().cpu().numpy()
    os.makedirs(os.path.dirname(embs_path), exist_ok=True)
    logger.info(f"Saving to {embs_path}, shape: {V_mets_np.shape}")
    np.save(embs_path, V_mets_np)


def test(model, history_list, test_list, num_ent, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_ent:     number of entity
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        logger.info(f"Load Model name: {model_name}. Using best epoch : {checkpoint['epoch']}")  # use best stat checkpoint
        logger.info("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph_rel_attr(num_nodes, num_ent, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels,
                                                                 test_triples_input, use_cuda)
        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score,
                                                                                        all_ans_r_list[time_idx],
                                                                                        eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score,
                                                                                all_ans_list[time_idx], eval_bz=1000,
                                                                                rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1

    mrr_raw = utils.stat_ranks(logger, ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(logger, ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(logger, ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(logger, ranks_filter_r, "filter_rel")

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    logger.info("loading graph data")
    data = utils.load_data(args.dataset)
    # 按照时间片划分训练集、验证集、测试集
    all_list = utils.split_by_time(data.all)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes,
                                                               False)  # entity prediction
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes,
                                                                 True)  # relation prediction
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = "{}-{}-{}-ly{}-hidden{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}-gpu{}" \
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.n_hidden, args.train_history_len,
                args.weight, args.discount, args.angle,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
    model_state_dir = f"../models/{args.dataset}/"
    os.makedirs(os.path.dirname(model_state_dir), exist_ok=True)
    model_state_file = model_state_dir + sanitize_filename(model_name) + ".pt"
    logger.info(f"Sanity Check: stat name : {model_state_file}")
    logger.info(f"Sanity Check: Is cuda available ? {torch.cuda.is_available()}")

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                          num_nodes,
                          num_rels,
                          args.n_hidden,
                          args.opn,
                          num_ent=args.entity_number,
                          num_rel=args.relation_number,
                          num_attr=args.attribute_number,
                          sequence_len=args.train_history_len,
                          num_bases=args.n_bases,
                          num_basis=args.n_basis,
                          num_hidden_layers=args.n_layers,
                          dropout=args.dropout,
                          self_loop=args.self_loop,
                          skip_connect=args.skip_connect,
                          layer_norm=args.layer_norm,
                          input_dropout=args.input_dropout,
                          hidden_dropout=args.hidden_dropout,
                          feat_dropout=args.feat_dropout,
                          aggregation=args.aggregation,
                          weight=args.weight,
                          discount=args.discount,
                          angle=args.angle,
                          entity_prediction=args.entity_prediction,
                          relation_prediction=args.relation_prediction,
                          use_cuda=use_cuda,
                          gpu=args.gpu,
                          analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    num_ent = args.entity_number
    if args.evolve and os.path.exists(model_state_file):
        embs_path = f"../embs/{args.dataset}/d{args.n_hidden}_l{args.n_layers}"
        evolve(model, all_list, num_ent, num_nodes, use_cuda, model_state_file, embs_path)
    elif args.evolve and not os.path.exists(model_state_file):
        logger.warning(f"--------------{model_state_file} not exist, Change mode to train and generate stat for evolve----------------\n")
    elif args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            train_list + valid_list,
                                                            test_list,
                                                            num_ent,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test,
                                                            all_ans_list_r_test,
                                                            model_state_file,
                                                            "test")
        return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    elif args.test and not os.path.exists(model_state_file):
        logger.warning(f"--------------{model_state_file} not exist, Change mode to train and generate stat for testing----------------\n")
    else:
        logger.info("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num + 1]
                if train_sample_num - args.train_history_len < 0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                            train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph_rel_attr(num_nodes, num_ent, snap, use_cuda, args.gpu) for snap in
                                 input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [
                    torch.from_numpy(_).long() for _ in output]
                loss_e, loss_r = model.get_loss(history_glist, output[0], use_cuda)
                loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            logger.info(
                "Epoch {:04d} | Ave Loss: {:.4f} | entity-relation:{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), best_mrr,
                        model_name))

            # validation
            if (epoch + 1) % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                    train_list,
                                                                    valid_list,
                                                                    num_ent,
                                                                    num_rels,
                                                                    num_nodes,
                                                                    use_cuda,
                                                                    all_ans_list_valid,
                                                                    all_ans_list_r_valid,
                                                                    model_state_file,
                                                                    mode="train")

                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            train_list + valid_list,
                                                            test_list,
                                                            num_ent,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test,
                                                            all_ans_list_r_test,
                                                            model_state_file,
                                                            mode="test")
        return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


# python main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attr_REGCN')

    # common parameters
    parser.add_argument("--gpu", type=int, default=0,  # -1
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,   # 0.7
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=True,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=True,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=5,  # 500
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help
         ="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=1,  # 20
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,  # 10
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=1,  # 20
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    # configuration for data config
    parser.add_argument("-d", "--dataset", type=str, default=dataset,
                        help="dataset to use")
    parser.add_argument("--n-hidden", type=int, default=cfg['n_hidden'],    # 8, 16, 24, 32, 40
                        help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=cfg['n_layers'],  # 1, 2, 3, 4
                        help="number of propagation rounds")
    parser.add_argument("--evolve", action='store_true', default=cfg['evolve'],  # False
                        help="load stat from dir and directly evolve")
    parser.add_argument("--test", action='store_true', default=cfg['test'],  # False
                        help="load stat from dir and directly test")
    parser.add_argument("--entity-number", type=int, default=cfg['entity_number'],
                        help="number of entity")
    parser.add_argument("--relation-number", type=int, default=cfg['relation_number'],
                        help="number of relation")
    parser.add_argument("--attribute-number", type=int, default=cfg['attribute_number'],
                        help="number of attribute")

    args = parser.parse_args()
    logger.info(args)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder + "-" + args.decoder)
        o_f = open(out_log, 'w')
        logger.info("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            logger.error("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        logger.info(f'* {len(grid)} hyperparameter combinations to try')
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            logger.info(f'* Hyperparameter Set {i}:')
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            logger.info(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr, hits, ranks = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            logger.info(f"MRR (raw): {mrr:.6f}")
            o_f.write("MRR (raw): {:.6f}\n".format(mrr))
            for hit in hits:
                avg_count = torch.mean((ranks <= hit).float())
                logger.info(f"Hits (raw) @ {hit}: {avg_count.item():.6f}")
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
    # single run
    else:
        run_experiment(args)
    sys.exit()