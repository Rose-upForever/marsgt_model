import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax as Softmax
from torchmetrics.functional import pairwise_cosine_similarity
import math

class GeneralConv(nn.Module):
    def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm=True):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        # Store attention matrix results
        self.res_att = None
        # Select different graph convolution layers based on the conv_name parameter
        if self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
    def forward(self, meta_xs, node_type, edge_index, edge_type):
        if self.conv_name == 'hgt':
            a = self.base_conv(meta_xs, node_type, edge_index, edge_type)
            self.res_att = self.base_conv.res_att
            return a
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'dense_hgt':
            return self.base_conv(meta_xs, node_type, edge_index, edge_type)

# HGTConv is the heterogeneous graph transformation convolution layer
# MessagePassing类是消息传递类，就是通过对图的计算图就算生成节点的嵌入表示。
class HGTConv(MessagePassing):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=True, distribution='uniform', **kwargs):
        super(HGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations #num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.distribution = distribution
        self.att = None
        self.res_att = None
        self.res = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        # Parameter用以定义一个可以被训练的参数矩阵
        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))  # TODO
        # 用以捕捉边信息的矩阵W
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        # glorot函数用于初始化权重参数
        glorot(self.relation_att)
        glorot(self.relation_msg)
    def initialize_weights(self):
        # self.modules()函数用于返回网络中的所有模块,
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                if self.distribution == 'uniform':
                    # xavier_uniform_函数用于初始化权重参数
                    torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if self.distribution == 'normal':
                    torch.nn.init.xavier_normal_(m.weight, gain=1)
    def forward(self, node_inp, node_type, edge_index, edge_type):
        # MeassagePassin 通过propagate完成，需要通过三个阶段：消息传递meassage, 消息聚合aggregate, 节点更新update
        # propagate会依次调用message, aggregate, update函数。
        # message是用来构建节点消息的
        # aggregate是用来聚合节点消息的
        # update是用来更新节点的
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, edge_type=edge_type)
    # 重写meassagepassing中的message和update函数
    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type):
        # 这里message中参数的后缀i和j是有原始限制的
        # 详解可以看  https://zhuanlan.zhihu.com/p/397560946
        '''
            j: source, i: target; <j, i>
            i：表示target节点的参数，j表示source节点的参数,源节点是信息的出发地，目标节点是信息的聚集地
        '''
        data_size = edge_index_i.size(0)

        self.res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        # 这里计算的注意力只是target与其邻居节点之间的注意力。
        for source_type in range(self.num_types):
            sb = (node_type_j == int(source_type)) # node_type_j 代表的源节点
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & tb  #idx is all the edges with meta relation <source_type, relation_type, target_type>
                    if idx.sum() == 0:
                        continue

                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]

                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    # torch.bmm实现矩阵的乘法
                    # 此处的操作用是论文中用以捕捉边信息的公式attention机制
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
                    self.res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk

                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)

        res = res_msg * Softmax(self.res_att, edge_index_i).view(-1, self.n_heads, 1)

        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))

            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        self.res = res
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)