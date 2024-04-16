#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 22:05
# @Author  : 我的名字
# @File    : test.py
# @Description : 这个函数是用来balabalabala自己写
import MarsGT
from MarsGT.conv import *
from MarsGT.egrn import *
from MarsGT.marsgt_model import *
from MarsGT.utils import *
import anndata as ad
from collections import Counter
import copy
import dill
from functools import partial
import json
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from operator import itemgetter
import random
import scipy.sparse as sp
from scipy.io import mmread
from scipy.sparse import hstack, vstack, coo_matrix
import seaborn as sb
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import SparsePCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import time
import torch
import torch.cuda as cuda
from torch import nn
from torch.autograd import Variable
import torch.distributions as D
import torch.nn.functional as F
import torch_geometric.data as Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax as Softmax
from torchmetrics.functional import pairwise_cosine_similarity
import warnings
from warnings import filterwarnings
import xlwt
import argparse
from tqdm import tqdm
import scanpy as sc
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix
if __name__ == "__main__":
    filterwarnings("ignore")
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Training GNN on gene cell graph')
    parser.add_argument('--fi', type=int,
                        default=0)  # This parameter is used for the benchmark to specify the starting sequence number of the created files
    parser.add_argument('--labsm', type=float, default=0.1)  # The rate of LabelSmoothing
    parser.add_argument('--wd', type=float,
                        default=0.1)  # The 'weight_decay' parameter is used to specify the strength of L2 regularization
    parser.add_argument('--lr', type=float, default=0.000001)  # learning rate
    parser.add_argument('--n_hid', type=int,
                        default=104)  # The number of layers should be a multiple of 'n_head' in order to make any modifications
    parser.add_argument('--nheads', type=int,
                        default=8)  # The 'heads' parameter represents the number of attention heads in the attention mechanism
    parser.add_argument('--nlayers', type=int, default=3)  # The number of layers in network
    parser.add_argument('--cell_size', type=int, default=30)  # The number of cells per subgraph (batch)
    parser.add_argument('--neighbor', type=int,
                        default=20)  # The number of neighboring nodes to be selected for each cell in the subgraph
    parser.add_argument('--egrn', type=bool, default=True)  # Whether to output the Enhancer-Gene regulatory network
    parser.add_argument('--epochs', type=int, default=3)  # The epoch number of NodeDimensionReduction
    parser.add_argument('--num_epochs', type=int, default=1)  # The epoch number of MarsGT-Model
    parser.add_argument('--output_file', type=str,
                        default='Tutorial_example\output')  # Please choose an output path to replace this path on your own.
    args = parser.parse_args([])

    # os.chdir('Tutorial_example') # Please replace the actual path with the path to "Tutorial_example" file.
    gene_peak = sparse.load_npz('D:\生信\代码及数据文件\marsgt\dataset\Gene_Peak_.npz')
    peak_cell = sparse.load_npz('D:\生信\代码及数据文件\marsgt\dataset\ATAC.npz')
    gene_cell = sparse.load_npz('D:\生信\代码及数据文件\marsgt\dataset\RNA.npz')
    true_label = np.load('D:\生信\代码及数据文件\marsgt\dataset\label500.npy', allow_pickle=True)
    gene_names = pd.DataFrame(np.load('D:\生信\代码及数据文件\marsgt\dataset\gene_name.npy', allow_pickle=True))
    peak_names = pd.DataFrame(np.load('D:\生信\代码及数据文件\marsgt\dataset\peak_name.npy', allow_pickle=True))
    # 生成的
    index = pd.read_csv('D:\生信\代码及数据文件\marsgt\dataset\index.txt')

    peak_cell.obs_names = peak_names[0]
    gene_cell.obs_names = gene_names[0]
    gene_peak.obs_names = gene_names[0]
    gene_peak.var_names = peak_names[0]

    RNA_matrix = gene_cell
    ATAC_matrix = peak_cell
    RP_matrix = gene_peak
    Gene_Peak = gene_peak

    cell_num = RNA_matrix.shape[1]
    gene_num = RNA_matrix.shape[0]
    peak_num = ATAC_matrix.shape[0]
    output_file = args.output_file
    fi = args.fi
    labsm = args.labsm
    lr = args.lr
    wd = args.wd
    n_hid = args.n_hid
    nheads = args.nheads
    nlayers = args.nlayers
    cell_size = args.cell_size
    neighbor = args.neighbor
    egrn = args.egrn
    epochs = args.epochs
    num_epochs = args.num_epochs
    device = torch.device("cpu")
    print('You will use : ', device)
    # 进行简单聚类得到聚类的数量
    # clustering result by scanpy
    initial_pre = initial_clustering(RNA_matrix)
    # number of every cluster
    cluster_ini_num = len(set(initial_pre))
    ini_p1 = [int(i) for i in initial_pre]

    # partite the data into batches
    # 与此同时，该函数还会选择每一个细胞节点的gene邻居和peak邻居。
    indices, Node_Ids, dic = batch_select_whole(RNA_matrix, ATAC_matrix, neighbor=[neighbor], cell_size=cell_size)
    n_batch = len(indices)

    # Reduce the dimensionality of features for cell, gene, and peak data.
    # num_realations代表的是图中关系的数量，也就是细胞和gene，peak之间的关系
    node_model = NodeDimensionReduction(RNA_matrix, ATAC_matrix, indices, index, ini_p1, n_hid=n_hid, n_heads=nheads,
                                        n_layers=nlayers, labsm=labsm, lr=lr, wd=wd, device=device, num_types=3,
                                        num_relations=6, epochs=100)
    # 此处得到cell_emb, gene_emb只是子图的
    # 通过子图训练得到的gnn，以便再整图中得到应用。
    # 此过程得到了在子图上训练得到的GNN
    gnn, cell_emb, gene_emb, peak_emb, h = node_model.train_model(n_batch=n_batch)

    # # Instantiate the MarsGT_model
    # MarsGT_model = MarsGT(gnn=gnn, h=h, labsm=labsm, n_hid=n_hid, n_batch=n_batch, device=device, lr=lr, wd=wd,
    #                       num_epochs=1)
    # # Train the model
    # MarsGT_gnn = MarsGT_model.train_model(indices=indices, RNA_matrix=RNA_matrix, ATAC_matrix=ATAC_matrix,
    # #                                       Gene_Peak=Gene_Peak, ini_p1=ini_p1,)
    # The result of MarsGT
    MarsGT_result = MarsGT_pred(RNA_matrix, ATAC_matrix, RP_matrix, egrn=False, MarsGT_gnn=gnn, indices=indices,
                                nodes_id=Node_Ids, cell_size=cell_size, device=device, gene_names=gene_names,
                                peak_names=peak_names)


    pred_label = MarsGT_result['pred_label']
    p_score, labels = purity_score(np.array(true_label), pred_label)
    e = Entropy(np.array(pred_label, dtype='int64'), np.array(labels, dtype='int64'))
    print("purity:%.4f" % p_score)
    print("NMI:%.4f" % normalized_mutual_info_score(true_label, labels))
    print("Entropy:%.4f" % e)
    # # Save numpy arrays to files
    # np.save(output_file + "/Node_Ids.npy", Node_Ids)
    # np.save(output_file + "/pred.npy", MarsGT_result['pred_label'])
    # np.save(output_file + "/cell_embedding.npy", MarsGT_result['cell_embedding'])
