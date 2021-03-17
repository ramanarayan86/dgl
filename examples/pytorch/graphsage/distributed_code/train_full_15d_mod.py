"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import argparse
import math
import os
import time
import numpy as np
import networkx as nx
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
#from dgl.nn.pytorch.conv import SAGEConvAgg, SAGEConvMLP
from sageconv import SAGEConvAgg, SAGEConvMLP
# import torch_ccl

import socket

def get_proc_groups(rank, size, replication):
    rank_c = rank // replication
     
    row_procs = []
    for i in range(0, size, replication):
        row_procs.append(list(range(i, i + replication)))

    col_procs = []
    for i in range(replication):
        col_procs.append(list(range(i, size, replication)))

    row_groups = []
    for i in range(len(row_procs)):
        row_groups.append(dist.new_group(row_procs[i]))

    col_groups = []
    for i in range(len(col_procs)):
        col_groups.append(dist.new_group(col_procs[i]))

    return row_groups, col_groups

class GraphSAGE(nn.Module):
  def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, aggregator_type, rank, size, replication, group, row_groups, col_groups):
    super(GraphSAGE, self).__init__()
    self.agg_layers = nn.ModuleList()
    self.mlp_layers = nn.ModuleList()
    self.dropout = nn.Dropout(dropout)
    self.activation = activation
    self.rank = rank
    self.replication = replication
    self.size = size
    self.group = group
    self.row_groups = row_groups
    self.col_groups = col_groups

    # input layer
    self.agg_layers.append(SAGEConvAgg(in_feats, aggregator_type))
    # hidden layers
    for i in range(n_layers - 1):
        self.agg_layers.append(SAGEConvAgg(n_hidden, aggregator_type))
    # output layer
    self.agg_layers.append(SAGEConvAgg(n_hidden, aggregator_type)) # activation None

    # input layer
    self.mlp_layers.append(SAGEConvMLP(in_feats, n_hidden, aggregator_type))
    # hidden layers
    for i in range(n_layers - 1):
        self.mlp_layers.append(SAGEConvMLP(n_hidden, n_hidden, aggregator_type))
    # output layer
    self.mlp_layers.append(SAGEConvMLP(n_hidden, n_classes, aggregator_type)) # activation None

  def forward(self, graph, inputs, ampbyp):
    h = self.dropout(inputs)
    for l, (agg_layer, mlp_layer) in enumerate(zip(self.agg_layers, self.mlp_layers)):
      z = agg_layer(self, graph, ampbyp, h)
      h = mlp_layer(graph, z, z)
    return h

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx, normalization):
    if not normalization:
        return adj_part

    adj_part = adj_part.coalesce()
    deg = torch.histc(adj_matrix[0].double(), bins=node_count)
    deg = deg.pow(-0.5)

    row_len = adj_part.size(0)
    col_len = adj_part.size(1)

    dleft = torch.sparse_coo_tensor([np.arange(0, row_len).tolist(),
                                     np.arange(0, row_len).tolist()],
                                     deg[row_vtx:(row_vtx + row_len)].float(),
                                     size=(row_len, row_len),
                                     requires_grad=False, device=torch.device("cpu"))

    dright = torch.sparse_coo_tensor([np.arange(0, col_len).tolist(),
                                     np.arange(0, col_len).tolist()],
                                     deg[col_vtx:(col_vtx + col_len)].float(),
                                     size=(col_len, col_len),
                                     requires_grad=False, device=torch.device("cpu"))
    # adj_part = torch.sparse.mm(torch.sparse.mm(dleft, adj_part), dright)
    ad_ind, ad_val = torch_sparse.spspmm(adj_part._indices(), adj_part._values(), 
                                            dright._indices(), dright._values(),
                                            adj_part.size(0), adj_part.size(1), dright.size(1))

    adj_part_ind, adj_part_val = torch_sparse.spspmm(dleft._indices(), dleft._values(), 
                                                        ad_ind, ad_val,
                                                        dleft.size(0), dleft.size(1), adj_part.size(1))

    adj_part = torch.sparse_coo_tensor(adj_part_ind, adj_part_val, 
                                                size=(adj_part.size(0), adj_part.size(1)),
                                                requires_grad=False, device=torch.device("cpu"))

    return adj_part

# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(adj_matrix, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero().squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero().squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

def oned_partition(rank, size, replication, inputs, adj_matrix, normalization):
    node_count = inputs.size(0)
    n_per_proc = math.ceil(float(node_count) / (size / replication))

    am_partitions = None
    am_pbyp = None

    rank_c = rank // replication
    # Compute the adj_matrix and inputs partitions for this process
    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)

        proc_node_count = vtx_indices[rank_c + 1] - vtx_indices[rank_c]
        am_pbyp, _ = split_coo(am_partitions[rank_c], node_count, n_per_proc, 0)
        for i in range(len(am_pbyp)):
            if i == size // replication - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank_c], normalization)
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank_c], normalization)

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1)), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count, 0, 
                                                vtx_indices[i], normalization)

        input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / (size / replication)), dim=0)

        adj_matrix_loc = am_partitions[rank_c]
        inputs_loc = input_partitions[rank_c]

    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs_loc.size: {inputs_loc.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp


def evaluate(model, graph, features, labels, nid, ampbyp):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features, ampbyp)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    # Initialize distributed environment with SLURM
    if "SLURM_PROCID" in os.environ.keys():
        os.environ["RANK"] = os.environ["SLURM_PROCID"]

    if "SLURM_NTASKS" in os.environ.keys():
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    print(f"hostname: {socket.gethostname()} rank: {os.environ['RANK']}", flush=True)
    os.environ["MASTER_ADDR"] = args.hostname 
    os.environ["MASTER_PORT"] = "1234"

    # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank) 

    # dist.init_process_group(backend='mpi')
    # dist.init_process_group(backend='gloo')
    dist.init_process_group(backend=args.dist_backend)
    rank = dist.get_rank()
    size = dist.get_world_size()
    print(f"hostname: {socket.gethostname()} rank: {rank} size: {size}")

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ.get("PMI_SIZE", -1))
    # if args.world_size == -1: args.world_size = int(os.environ["WORLD_SIZE"])

    # args.distributed = args.world_size > 1
    # if args.distributed:
    #     args.rank = int(os.environ.get("PMI_RANK", -1))
    #     if args.rank == -1: args.rank = int(os.environ["RANK"])        
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)

    #     print("Rank: ", args.rank ," World_size: ", args.world_size)

    # rank = args.rank
    # size = args.world_size

    print("Processes: " + str(size))
    device = torch.device('cpu')

    group = dist.new_group(list(range(size)))
    row_groups, col_groups = get_proc_groups(rank, size, args.replication)

    # Partition input graph and features
    edges = g.all_edges() # Convert DGLGraph to COO for partitioning
    edges = torch.stack([edges[0], edges[1]], dim=0)
    features_loc, g_loc, ampbyp = oned_partition(rank, size, args.replication, features, edges, 
                                                        args.normalization)
    # print("------------------->",ampbyp)
    # Convert COO back to DGLGraph
    # Uses hardcoded types, doesn't include all the metadata in original g
    # g_loc = dgl.heterograph({("_N", "_N", "_E"): (g_loc._indices()[0], g_loc._indices()[1])}) 

    for i in range(len(ampbyp)):
        ampbyp[i] = ampbyp[i].t().coalesce().to(device)
        print(f"i: {i} ampbyp.size: {ampbyp[i].size()}")
        # ampbyp[i] = dgl.heterograph({("_N", "_N", "_E"): (ampbyp[i]._indices()[0], ampbyp[i]._indices()[1])})
        # ampbyp[i] = dgl.graph((ampbyp[i]._indices()[0], ampbyp[i]._indices()[1]))
        # ampbyp[i] = dgl.heterograph({am_pbyp[i]})

    features.requires_grad = True
    # create GraphSAGE model
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type,
                      rank,
                      size,
                      args.replication,
                      group,
                      row_groups,
                      col_groups)

    if cuda:
        model.cuda()

    print(f"is cuda: {next(model.parameters()).is_cuda}:")
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        # logits = model(g, features, ampbyp)
        logits = model(g_loc, features_loc, ampbyp)
        
        rank_c = rank // args.replication
        var = logits.size(0)
        rank_train_mask = torch.split(train_mask, logits.size(0), dim=0)[rank_c]
        rank_val_mask = torch.split(val_mask, logits.size(0), dim=0)[rank_c]
        label_rank = torch.split(labels, logits.size(0), dim=0)[rank_c]
        train_nids= rank_train_mask.nonzero().squeeze()
        val_nids = rank_val_mask.nonzero().squeeze()

        print("train_mask ==========", train_mask.shape)
        print("labl =============", labels.dtype)
        print("rank_train_mask ===========", train_nids.dtype)
        print("label_rank =============", label_rank.dtype)
        print("labels ------>",label_rank[rank_train_mask].size()) #, " ", labels[train_nid].shape)
        print("logits ------>",logits[rank_train_mask].size())
         
        # if list(label_rank[rank_train_mask].size())[0] > 0:
        # loss = F.cross_entropy(logits[rank_train_mask], label_rank[rank_train_mask]) 
        # for param in model.parameters(): param.requires_grad = True 
        # for param in model.parameters(): param.requires_grad=True
        loss = F.cross_entropy(logits[train_nids], label_rank[train_nids]) 
        # loss = Variable(loss, requires_grad = True)
        # loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        
        # else:
        #     fake_loss = (logits * torch.FloatTensor(logits.size(), device=device).fill_(0)).sum()
        #     fake_loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, g, features, labels, val_nids, ampbyp)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    rank_test_mask = torch.split(test_mask, logits.size(0), dim=0)[rank_c]
    test_nids = rank_test_mask.nonzero().squeeze()
    print()
    acc = evaluate(model, g, features, labels, test_nids, ampbyp)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--replication", type=int, default=1,
                        help="replciation factor for 1.5D algorithm")
    parser.add_argument("--normalization", type=bool, default=False,
                        help="normalize adjacency matrix for training")
    parser.add_argument('--world-size', default=-1, type=int,
                         help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                         help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                            help='distributed backend')
    parser.add_argument('--hostname', default='127.0.0.1', type=str,
                            help='hostname for rank 0')
    args = parser.parse_args()
    print(args)

    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ.get("PMI_SIZE", -1))
    #     if args.world_size == -1: args.world_size = int(os.environ["WORLD_SIZE"])

    # args.distributed = args.world_size > 1
    # if args.distributed:
    #     args.rank = int(os.environ.get("PMI_RANK", -1))
    #     if args.rank == -1: args.rank = int(os.environ["RANK"])        
    #     dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)

    # print("Rank: ", args.rank ," World_size: ", args.world_size)

    # rank = args.rank
    # size = args.world_size

    # print("Processes: " + str(size))
    # device = torch.device('cpu')

    main(args)
