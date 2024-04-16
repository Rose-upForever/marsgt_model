import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import *
from .utils import *
from .egrn import *
# from tensorboardX import SummaryWriter
# Sumwrite = SummaryWriter(log_dir='../Data/log')
class GNN_from_raw(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, conv_name='hgt',
                 prev_norm=True, last_norm=True):
        # super函数主要是调用父类中的方法，这里是调用了父类nn.Module中的__init__()方法
        super(GNN_from_raw, self).__init__()
        # ModuleList是一个储存不同模块的列表,一个模块可以是一个层
        self.gcs = nn.ModuleList()
        # num_types指的是图中的节点类型
        self.num_types = num_types
        # in_dim = [RNA_matrix.shape[0], RNA_matrix.shape[1], ATAC_matrix.shape[1]]
        # indim分别代表的是图中三个节点的特征原始维度
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.embedding1 = nn.ModuleList()

        # Initialize MLP weight matrices
        # 对于图中三个不同类型的节点，构建三个全连接网络用以降维操作。
        for ti in range(num_types):
            self.embedding1.append(nn.Linear(in_dim[ti], 256))

        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(256, n_hid))

        # Initialize graph convolution layers
        # gcs也就是构建的tansformer的模块，transformer由多个模块堆叠而成，这里的n_layers类似于多个自编码器的设置
        for l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=prev_norm))
        self.gcs.append(
            GeneralConv(conv_name, n_hid, n_hid, num_types, num_relations, n_heads, dropout, use_norm=last_norm))

    def encode(self, x, t_id):
        h1 = F.relu(self.embedding1[t_id](x))
        return h1
    # 生成节点的嵌入表示
    def forward(self, node_feature, node_type, edge_index, edge_type):
        node_embedding = []
        for t_id in range(self.num_types):
            node_embedding += list(self.encode(node_feature[t_id], t_id))
        # 在MarsGT模型建立图的时候之所以行索引前额外加上一个数，就是因为此处进行了嵌入的叠加。
        #np.nonzero(gene_cell_sub)[0]取得是非0元素中基因的索引，但是在node_feature中，gene的位置在cell的后面，所以需要加上cell的数量gene_cell_sub.shape[1]
        node_embedding = torch.stack(node_embedding)
        # Initialize result matrix
        res = torch.zeros(node_embedding.size(0), self.n_hid).to(node_feature[0].device)

        # Process each node type
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            # Update result matrix
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_embedding[idx]))

        # Apply dropout to the result matrix
        # dropout通常添加在全连接层之后，以表示随机丢弃一些神经元 # adapt_ws的作用是将原始数据映射为n_hid维度，其中n_hid = n_head* d_k
        meta_xs = self.drop(res)
        del res

        # Iterate through graph convolution layers and update result matrix
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type)

        return meta_xs


class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x


class NodeDimensionReduction(nn.Module):
    def __init__(self, RNA_matrix, ATAC_matrix, indices, index, ini_p1, n_hid, n_heads,
                 n_layers, labsm, lr, wd, device, num_types=3, num_relations=4, epochs=1):
        super(NodeDimensionReduction, self).__init__()
        self.RNA_matrix = RNA_matrix
        self.ATAC_matrix = ATAC_matrix
        self.indices = indices
        self.ini_p1 = ini_p1
        self.in_dim = [RNA_matrix.shape[0], RNA_matrix.shape[1], ATAC_matrix.shape[1]]
        self.n_hid = n_hid
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.labsm = labsm
        self.lr = lr
        self.wd = wd
        self.index = index
        self.device = device
        self.epochs = epochs
        # LabelSmoothing函数的主要作用是为了减少过拟合的风险
        self.LabSm = LabelSmoothing(self.labsm)
        # gnn构建起的是基于transformer的图神经网络框架
        self.gnn = GNN_from_raw(in_dim=self.in_dim,
                                n_hid=self.n_hid,
                                num_types=self.num_types,
                                num_relations=self.num_relations,
                                n_heads=self.n_heads,
                                n_layers=self.n_layers,
                                dropout=0.3).to(self.device)
        # gnn.parameters()返回的是模型中可学习的参数，即权重和偏置,即优化其中的参数
        self.optimizer = torch.optim.RMSprop(self.gnn.parameters(), lr=self.lr, weight_decay=self.wd)
        # 这个函数是一个类方法，用于创建一个学习率衰减调度器。它将与一个优化器对象关联，并在验证损失停滞不前时将学习率减少一半。
        # 它设置了一个耐心参数，表示在多少次验证中损失都没有改善才会触发学习率衰减，以及一个模式参数，表示是找最小损失还是最大损失
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5,
                                                                    verbose=True)

    def train_model(self, n_batch):
        print('The training process for the NodeDimensionReduction model has started. Please wait.')
        for epoch in tqdm(range(self.epochs)):
            for batch_id in np.arange(n_batch):
                gene_index = self.indices[batch_id]['gene_index']
                cell_index = self.indices[batch_id]['cell_index']
                peak_index = self.indices[batch_id]['peak_index']
                gene_feature = self.RNA_matrix[list(gene_index),]
                cell_feature = self.RNA_matrix[:, list(cell_index)].T
                peak_feature = self.ATAC_matrix[list(peak_index),]
                index = self.index
                index_gene = pd.merge(index,pd.DataFrame(gene_index,columns=['gene_index']),how='inner',on='gene_index')
                index_peak = pd.merge(index_gene, pd.DataFrame(peak_index, columns=['peak_index']), how='inner',on='peak_index')
                gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(self.device)
                cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(self.device)
                peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(self.device)

                node_feature = [cell_feature, gene_feature, peak_feature]
                gene_cell_sub = self.RNA_matrix[list(gene_index),][:, list(cell_index)]
                peak_cell_sub = self.ATAC_matrix[list(peak_index),][:, list(cell_index)]
                # gene_cell_edge_index = torch.LongTensor([np.nonzero(gene_cell_sub)[0]+gene_cell_sub.shape[1],np.nonzero(gene_cell_sub)[1]]).to(device)
                # peak_cell_edge_index = torch.LongTensor([np.nonzero(peak_cell_sub)[0]+gene_cell_sub.shape[0]+gene_cell_sub.shape[1],np.nonzero(peak_cell_sub)[1]]).to(device)

                # nonzero函数返回的是非零元素的索引,返回的形式是两个列表,第一个列表是行索引,第二个列表是列索引，形式类似于 (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
                gene_cell_edge_index1 = list(np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1]) + list(
                    np.nonzero(gene_cell_sub)[1])
                gene_cell_edge_index2 = list(np.nonzero(gene_cell_sub)[1]) + list(
                    np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1])
                gene_cell_edge_index = torch.LongTensor([gene_cell_edge_index1, gene_cell_edge_index2]).to(self.device)
                peak_cell_edge_index1 = list(
                    np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]) + list(
                    np.nonzero(peak_cell_sub)[1])
                peak_cell_edge_index2 = list(np.nonzero(peak_cell_sub)[1]) + list(
                    np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1])
                peak_cell_edge_index = torch.LongTensor([peak_cell_edge_index1, peak_cell_edge_index2]).to(self.device)

                #TODO:通过get_index 函数已经获得了存在关系的peak和gene在gene_sub/peak_sub中的索引，下一步就是如何利用这个索引在原图中添加gene和peak之间边。
                peak_index_Con = get_index(peak_cell_sub,peak_index,index_peak['peak_index'])
                gene_index_Con = get_index(gene_cell_sub,gene_index,index_peak['gene_index'])
                gene_index_Con = [i + gene_cell_sub.shape[1] for i in gene_index_Con]
                peak_index_Con = [j + gene_cell_sub.shape[1]+gene_cell_sub.shape[0] for j in peak_index_Con]
                gene_peak_edge_index1 = gene_index_Con + peak_index_Con
                gene_peak_edge_index2 = peak_index_Con + gene_index_Con
                gene_peak_edge_index = torch.LongTensor([gene_peak_edge_index1, gene_peak_edge_index2]).to(self.device)
                edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index,gene_peak_edge_index), dim=1)
                node_type = torch.LongTensor(np.array(
                    list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
                        np.ones(len(peak_index)) * 2))).to(self.device)
                # edge_type = torch.LongTensor(np.array(list(np.zeros(gene_cell_edge_index.shape[1]))+list(np.ones(peak_cell_edge_index.shape[1]) ))).to(device)
                # 定义四种边的类型：gene-cell，cell——gene，peak-cell,cell-peak
                edge_type = torch.LongTensor(np.array(list(np.zeros(np.nonzero(gene_cell_sub)[0].shape[0])) + list(
                    np.ones(np.nonzero(gene_cell_sub)[1].shape[0])) + list(
                    2 * np.ones(np.nonzero(peak_cell_sub)[0].shape[0])) + list(
                    3 * np.ones(np.nonzero(peak_cell_sub)[1].shape[0])) + list(
                    4*np.ones(len(gene_index_Con))) + list(5*np.ones(len(peak_index_Con))))).to(self.device)
                l = torch.LongTensor(np.array(self.ini_p1)[[cell_index]]).to(self.device)
                node_rep = self.gnn.forward(node_feature, node_type,
                                            edge_index,
                                            edge_type).to(self.device)
                cell_emb = node_rep[node_type == 0]
                gene_emb = node_rep[node_type == 1]
                peak_emb = node_rep[node_type == 2]
                # mm用以执行两个矩阵之间的乘法操作，通过该操作可以生成gene_cell和gene_peak矩阵
                decoder1 = torch.mm(gene_emb, cell_emb.t())
                decoder2 = torch.mm(peak_emb, cell_emb.t())
                gene_cell_sub = torch.tensor(np.array(gene_cell_sub.todense()), dtype=torch.float32).to(self.device)
                peak_cell_sub = torch.tensor(np.array(peak_cell_sub.todense()), dtype=torch.float32).to(self.device)

                logp_x1 = F.log_softmax(decoder1, dim=-1)
                p_y1 = F.softmax(gene_cell_sub, dim=-1)
                # 计算生成的矩阵和原始之间的KL散度，也就是相似度
                loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='mean')

                logp_x2 = F.log_softmax(decoder2, dim=-1)
                p_y2 = F.softmax(peak_cell_sub, dim=-1)

                loss_kl2 = F.kl_div(logp_x2, p_y2, reduction='mean')
                loss_kl = loss_kl1 + loss_kl2

                loss_cluster = self.LabSm(cell_emb, l)
                lll = 0
                l = l.view(-1).tolist()
                g = [int(i) for i in l]
                for i in set([int(k) for k in l]):
                    h = cell_emb[[True if i == j else False for j in g]]
                    ll = F.cosine_similarity(h[list(range(h.shape[0])) * h.shape[0],],
                                             h[[v for v in range(h.shape[0]) for i in range(h.shape[0])]]).mean()
                    lll = ll + lll
                loss = loss_cluster - lll
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print('The training for the NodeDimensionReduction model has been completed.')
        return self.gnn, cell_emb, gene_emb, peak_emb, h


class MarsGT(nn.Module):
    def __init__(self, gnn, h, labsm, n_hid, n_batch, device, lr, wd, num_epochs=1):
        super(MarsGT, self).__init__()
        self.lr = lr
        self.wd = wd
        self.gnn = gnn
        self.h = h
        self.n_hid = n_hid
        self.n_batch = n_batch
        self.device = device
        self.num_epochs = num_epochs
        self.net = Net(2 * self.n_hid, self.n_hid).to(self.device)
        self.gnn_optimizer = torch.optim.RMSprop(self.gnn.parameters(), lr=self.lr, weight_decay=self.wd)
        self.net_optimizer = torch.optim.RMSprop(self.net.parameters(), lr=1e-2)
        self.labsm = labsm
        self.LabSm = LabelSmoothing(self.labsm)

    def forward(self, indices, RNA_matrix, ATAC_matrix, Gene_Peak, ini_p1):
        cluster_l = list()
        cluster_kl_l = list()
        sim_l = list()
        nmi_l = list()
        ini_p1 = np.array(ini_p1)

        for epoch in range(self.num_epochs):
            for batch_id in tqdm(np.arange(self.n_batch)):
                gene_index = indices[batch_id]['gene_index']
                cell_index = indices[batch_id]['cell_index']
                peak_index = indices[batch_id]['peak_index']
                gene_feature = RNA_matrix[list(gene_index),]
                cell_feature = RNA_matrix[:, list(cell_index)].T
                peak_feature = ATAC_matrix[list(peak_index),]
                gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(self.device)
                cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(self.device)
                peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(self.device)
                node_feature = [cell_feature, gene_feature, peak_feature]
                gene_cell_sub = RNA_matrix[list(gene_index),][:, list(cell_index)]
                peak_cell_sub = ATAC_matrix[list(peak_index),][:, list(cell_index)]

                # 由于node_feature的顺序是cell,gene,peak
                # np.nonzero(gene_cell_sub)[0]取得是非0元素中基因的索引，但是在node_feature中，gene的位置在cell的后面，所以需要加上cell的数量gene_cell_sub.shape[1]
                gene_cell_edge_index1 = list(np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1]) + list(
                    np.nonzero(gene_cell_sub)[1])
                gene_cell_edge_index2 = list(np.nonzero(gene_cell_sub)[1]) + list(
                    np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1])
                gene_cell_edge_index = torch.LongTensor([gene_cell_edge_index1, gene_cell_edge_index2]).to(self.device)
                # 取peak的feature时，由于其在node_feature的最后面，所以要加上gene和cell的数量
                peak_cell_edge_index1 = list(
                    np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]) + list(
                    np.nonzero(peak_cell_sub)[1])
                peak_cell_edge_index2 = list(np.nonzero(peak_cell_sub)[1]) + list(
                    np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1])
                peak_cell_edge_index = torch.LongTensor([peak_cell_edge_index1, peak_cell_edge_index2]).to(self.device)
                # gene_cell_edge_index = torch.LongTensor([np.nonzero(gene_cell_sub)[0]+gene_cell_sub.shape[1],np.nonzero(gene_cell_sub)[1]]).to(device)
                # peak_cell_edge_index = torch.LongTensor([np.nonzero(peak_cell_sub)[0]+gene_cell_sub.shape[0]+gene_cell_sub.shape[1],np.nonzero(peak_cell_sub)[1]]).to(device)
                # TODO:gene_peak之间的关系在此处添加



                edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
                node_type = torch.LongTensor(np.array(
                    list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
                        np.ones(len(peak_index)) * 2))).to(self.device)
                edge_type = torch.LongTensor(np.array(list(np.zeros(np.nonzero(gene_cell_sub)[0].shape[0])) + list(
                    np.ones(np.nonzero(gene_cell_sub)[1].shape[0])) + list(
                    2 * np.ones(np.nonzero(peak_cell_sub)[0].shape[0])) + list(
                    3 * np.ones(np.nonzero(peak_cell_sub)[1].shape[0])))).to(self.device)

                # edge_type = torch.LongTensor(np.array(list(np.zeros(gene_cell_edge_index.shape[1]))+list(np.ones(peak_cell_edge_index.shape[1]) ))).to(device)
                l = torch.LongTensor(np.array(ini_p1)[[cell_index]]).to(self.device)
                # l2 = torch.LongTensor(label[[cell_index]])

                # l = torch.LongTensor(ini_p1)[[cell_index]].to(device)

                node_rep = self.gnn.forward(node_feature, node_type,
                                            edge_index,
                                            edge_type).to(self.device)
                cell_emb = node_rep[node_type == 0]
                gene_emb = node_rep[node_type == 1]
                peak_emb = node_rep[node_type == 2]

                decoder1 = torch.mm(gene_emb, cell_emb.t())
                decoder2 = torch.mm(peak_emb, cell_emb.t())
                gene_cell_sub = torch.tensor(np.array(gene_cell_sub.todense()), dtype=torch.float32).to(self.device)
                peak_cell_sub = torch.tensor(np.array(peak_cell_sub.todense()), dtype=torch.float32).to(self.device)

                logp_x1 = F.log_softmax(decoder1, dim=-1)
                p_y1 = F.softmax(gene_cell_sub, dim=-1)

                loss_kl1 = F.kl_div(logp_x1, p_y1, reduction='mean')

                logp_x2 = F.log_softmax(decoder2, dim=-1)
                p_y2 = F.softmax(peak_cell_sub, dim=-1)

                loss_kl2 = F.kl_div(logp_x2, p_y2, reduction='mean')

                loss_kl = loss_kl1 + loss_kl2

                lll2 = 0
                l_copy = l.view(-1).tolist()
                g = [int(i) for i in l_copy]
                for i in set([int(k) for k in l_copy]):
                    ll2 = F.cosine_similarity(self.h[list(range(self.h.shape[0])) * self.h.shape[0],], self.h[
                        [v for v in range(self.h.shape[0]) for i in range(self.h.shape[0])]]).mean()
                    lll2 = ll2 + lll2

                loss_cluster = self.LabSm(cell_emb, l)

                m = range(peak_emb.shape[0])
                gene_emb_enh = gene_emb[list(range(gene_emb.shape[0])) * peak_emb.shape[0]]
                peak_emb_enh = peak_emb[[v for v in m for i in range(gene_emb.shape[0])]]

                net_input = torch.cat((gene_emb_enh, peak_emb_enh), 1)
                gene_peak_cluster_pre = self.net(net_input)
                m = range(peak_feature.shape[0])
                # gene_feature_ori = gene_feature[list(range (gene_feature.shape[0]))*peak_emb.shape[0]]
                # peak_feature_ori = peak_feature[[v for v in m for i in range(gene_feature.shape[0])]]
                peak_feature_ori = peak_Sparse(peak_feature[:,cell_index],gene_feature[:,cell_index],self.device)
                gene_feature_ori = gene_Sparse(peak_feature[:,cell_index],gene_feature[:,cell_index],self.device)
                # g1p1 g2p1 g3p1
                gene_peak_ori = gene_feature_ori.mul(peak_feature_ori)
                gene_peak_1 = Gene_Peak[list(gene_index),][:, list(peak_index)].reshape(
                    len(gene_index) * len(peak_index), 1)
                row_ind = torch.Tensor(list(gene_peak_1.tocoo().row) * len(cell_index))
                col_ind = torch.Tensor([v for v in range(len(cell_index)) for i in range(len(gene_peak_1.tocoo().row))])
                data = torch.Tensor(list(gene_peak_1.tocoo().data) * len(cell_index))
                a = torch.sparse.FloatTensor(torch.vstack((row_ind, col_ind)).long(), data,
                                             (gene_peak_1.shape[0], len(cell_index))).to(self.device)
                gene_peak_cell = gene_peak_ori.mul(a)
                # the original in cell level (peak * gene * peak_gene)
                gene_peak_cell = gene_peak_ori.mul(a)
                # the original in cell cluster level (peak * gene * peak_gene)
                gene_peak_cell_cluster = torch.mm(gene_peak_ori.mul(a), cell_emb)

                logp_x3 = F.log_softmax(gene_peak_cluster_pre, dim=-1)
                p_y3 = F.softmax(gene_peak_cell_cluster, dim=-1)

                loss_net = F.kl_div(logp_x3, p_y3, reduction='mean')

                loss = loss_net + loss_cluster + loss_kl  # - lll2
                # loss = loss_cluster  + loss_kl - lll + loss_net  #+ loss_S_R#+ loss_net

                # print('=================================================loss=================================================')
                # print(loss)
                # print('=================================================grad=================================================')
                # for name, param in self.gnn.named_parameters():
                #     print(f"Gradient of {name}: {param.grad}")

                self.gnn_optimizer.zero_grad()
                self.net_optimizer.zero_grad()
                loss.backward()
                self.gnn_optimizer.step()
                self.net_optimizer.step()
        return self.gnn

    def train_model(self, indices, RNA_matrix, ATAC_matrix, Gene_Peak, ini_p1):
        self.train()
        print('The training process for the MarsGT model has started. Please wait.')
        Mars_gnn = self.forward(indices, RNA_matrix, ATAC_matrix, Gene_Peak, ini_p1)
        print('The training for the MarsGT model has been completed.')
        return Mars_gnn


def MarsGT_pred(RNA_matrix, ATAC_matrix, RP_matrix, egrn, MarsGT_gnn, indices, nodes_id, cell_size, device, gene_names,
                peak_names):
    n_batch = math.ceil(nodes_id.shape[0] / cell_size)
    embedding = []
    l_pre = []
    MarsGT_result = {}
    with torch.no_grad():
        for batch_id in tqdm(range(n_batch)):
            gene_index = indices[batch_id]['gene_index']
            cell_index = indices[batch_id]['cell_index']
            peak_index = indices[batch_id]['peak_index']
            gene_feature = RNA_matrix[list(gene_index),]
            cell_feature = RNA_matrix[:, list(cell_index)].T
            peak_feature = ATAC_matrix[list(peak_index),]
            gene_feature = torch.tensor(np.array(gene_feature.todense()), dtype=torch.float32).to(device)
            cell_feature = torch.tensor(np.array(cell_feature.todense()), dtype=torch.float32).to(device)
            peak_feature = torch.tensor(np.array(peak_feature.todense()), dtype=torch.float32).to(device)
            node_feature = [cell_feature, gene_feature, peak_feature]
            gene_cell_sub = RNA_matrix[list(gene_index),][:, list(cell_index)]
            peak_cell_sub = ATAC_matrix[list(peak_index),][:, list(cell_index)]
            gene_cell_edge_index = torch.LongTensor(
                [np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1], np.nonzero(gene_cell_sub)[1]]).to(device)
            peak_cell_edge_index = torch.LongTensor(
                [np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1],
                 np.nonzero(peak_cell_sub)[1]]).to(device)
            edge_index = torch.cat((gene_cell_edge_index, peak_cell_edge_index), dim=1)
            node_type = torch.LongTensor(np.array(
                list(np.zeros(len(cell_index))) + list(np.ones(len(gene_index))) + list(
                    np.ones(len(peak_index)) * 2))).to(device)
            edge_type = torch.LongTensor(np.array(
                list(np.zeros(gene_cell_edge_index.shape[1])) + list(np.ones(peak_cell_edge_index.shape[1])))).to(
                device)
            for name,param in MarsGT_gnn.named_parameters():
                print(name,':',param)
            node_rep = MarsGT_gnn.forward(node_feature, node_type,
                                          edge_index,
                                          edge_type).to(device)
            cell_emb = node_rep[node_type == 0]
            gene_emb = node_rep[node_type == 1]
            peak_emb = node_rep[node_type == 2]

            # If the device is CUDA, copy the tensor to CPU memory
            if device.type == "cuda":
                cell_emb = cell_emb.cpu()
            # It is now safe to convert the tensor to a NumPy array
            embedding.append(cell_emb.detach().numpy())

            cell_pre = list(cell_emb.argmax(dim=1).detach().numpy())
            l_pre.extend(cell_pre)

    cell_embedding = np.vstack(embedding)
    cell_clu = np.array(l_pre)

    if egrn:
        final_egrn_df = egrn_calculate(cell_clu, nodes_id, RNA_matrix, ATAC_matrix, RP_matrix, gene_names, peak_names)
        MarsGT_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding, 'egrn': final_egrn_df}
        return MarsGT_result
    else:
        MarsGT_result = {'pred_label': cell_clu, 'cell_embedding': cell_embedding}
        return MarsGT_result