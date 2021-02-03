import os
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from functools import wraps
from dgl.data import EDADataset
import tqdm
import traceback
from datetime import datetime
import pcl_mlp
try:
    import torch_ccl
except ImportError as e:
    print(e)

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        #self.dropout = nn.Dropout(dropout)
        self.dropout = pcl_mlp.Dropout(dropout)

        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int()
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels_or_slacks, datapins, mask, batch_size, device, args):
    """
    Evaluate the model on the validation set specified by ``mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels_or_slacks : The labels of all the nodes or slacks associated with them.
    mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """

    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    if args.task == "classification":
      pred = th.gather(th.Tensor(pred), 0, th.Tensor(datapins))
      labels_or_slacks = th.gather(th.Tensor(labels_or_slacks), 0, th.LongTensor(datapins))
      return compute_acc(pred, labels_or_slacks)
    else:

      batch_slacks = th.gather(th.Tensor(labels_or_slacks), 0, th.LongTensor(datapins))
      pred = th.gather(th.Tensor(pred), 0, th.LongTensor(datapins))

      cells = len(datapins)
      tns_pred = 0
      tns_act = 0
      tp = 0
      fp = 0
      tn = 0
      fn = 0

      for i in range(cells):
        if (batch_slacks[i] < 0):
          tns_act += batch_slacks[i]

      for i in range(cells):
        if (pred[i] < 0 and batch_slacks[i] < 0):
          tp += 1
          tns_pred += pred[i]
        elif (pred[i] < 0 and batch_slacks[i] > 0):
          fp += 1
        elif (pred[i] > 0 and batch_slacks[i] < 0):
          fn += 1
        elif (pred[i] > 0 and batch_slacks[i] > 0):
          tn += 1

      acc = (cells - (fp + fn))/cells
      slack_gap = (abs(tns_act) - abs(tns_pred))/abs(tns_act)

      return tns_pred, tns_act, tp, fp, tn, fn, acc, slack_gap

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    th.save(state, filename, _use_new_zipfile_serialization=True)

def adjust_learning_rate(optimizer, epoch, epoch_steps, args):
  """Sets the learning rate to the initial LR decayed by 10% every 1000 epochs"""
  lr = args.lr * (1. / (1.1 ** (epoch // epoch_steps)))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

#### Entry point
def run(args, device, data):
    if args.distributed:
      if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ.get("PMI_RANK", -1))
        if args.rank == -1: args.rank = int(os.environ["RANK"])
      dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # Unpack data
    if args.task == 'classification':
      if args.resume == '':
        train_mask, test_mask, in_feats, labels, n_classes, datapins, datapin_slacks, g = data
      else:
        train_mask, test_mask, labels, n_classes, datapins, datapin_slacks, g = data
    else:
      if args.resume == '':
        train_mask, in_feats, slacks, n_slacks, datapins, datapin_slacks, g = data
      else:
        train_mask, slacks, n_slacks, datapins, datapin_slacks, g = data

    train_nid = np.nonzero(train_mask)[0].astype(np.int64)
    train_nid = th.LongTensor(train_nid)
    if not args.all_train:
      test_nid = np.nonzero(test_mask)[0].astype(np.int64)
      test_nid = th.LongTensor(test_nid)

    if args.distributed:
      train_nid = th.split(train_nid, len(train_nid) // args.world_size)[args.rank]

    # Create sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # optionally resume from a checkpoint
    if args.resume:
      if os.path.isfile(args.resume):
        print("=> loading features checkpoint '{}'".format(args.resume))
        checkpoint = th.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        g.ndata['features'] = checkpoint['features']
        in_feats = g.ndata['features'].shape[1]
        print("=> loaded features checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
      else:
       print("=> no checkpoint found at '{}'".format(args.resume))

    # Define model and optimizer
    if args.task == "regression":
      model = SAGE(in_feats, args.num_hidden, n_slacks, args.num_layers, F.relu, args.dropout)
    else:
      model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)

    if args.distributed:
      model = th.nn.parallel.DistributedDataParallel(model)

    loss_fcn = nn.CrossEntropyLoss()
    if args.task == "regression":
      loss_fcn = nn.L1Loss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.resume:
      if args.distributed:
        if os.path.isfile(args.resume):
          print("=> loading model & optimizer checkpoint '{}'".format(args.resume))
          model.module.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          print("=> loaded model & optimizer checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
      else:
        if os.path.isfile(args.resume):
          print("=> loading model & optimizer checkpoint '{}'".format(args.resume))
          model.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          print("=> loaded model & optimizer checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    if args.task == 'regression':
      #Compute tns
      tns_act = th.zeros(1)
      datapin_slacks = th.Tensor(datapin_slacks)
      for i in range(len(datapins)):
        if (datapin_slacks[i] < 0):
          tns_act += datapin_slacks[i]
      
    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.start_epoch, args.num_epochs):
        maetrain = []
        adjust_learning_rate(optimizer, epoch, 1500, args)

        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = th.empty()
            batch_slacks = th.empty()
            if args.task == "classification":
              batch_labels = blocks[-1].dstdata['labels']
            else:
              batch_slacks = blocks[-1].dstdata['slacks']

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            if args.task == "classification":
              loss = loss_fcn(batch_pred, batch_labels)
            else:
              loss = loss_fcn(batch_pred, batch_slacks)
            optimizer.zero_grad()
            loss.backward()

            if args.distributed:
              for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                  th.distributed.all_reduce(param.grad.data, op=th.distributed.ReduceOp.SUM)

            optimizer.step()

            if args.task == "regression":
              if not args.distributed or (args.distributed and args.rank == 0): 
                maetrain.append(F.l1_loss(batch_pred, batch_slacks).cpu().item())

            if not args.distributed or (args.distributed and args.rank == 0): 
              iter_tput.append(len(seeds) * args.world_size / (time.time() - tic_step))

            if not args.distributed or (args.distributed and args.rank == 0): 
              if step % args.log_every == 0:
                if args.task == 'classification':
                  acc = compute_acc(batch_pred[batch_label_nids], batch_labels)
                  print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Time {:.4f} | Speed (samples/sec) {:.4f}'.format(
                    epoch, step, loss.item(), acc.item(), (time.time() - tic_step), np.mean(iter_tput[3:])))
                else:
                  print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train MAE {:.4f} | Time {:.4f} | Speed (samples/sec) {:.4f}'.format(
                    epoch, step, loss.item(), np.mean(maetrain), (time.time() - tic_step), np.mean(iter_tput[3:])))

        if args.distributed:
          th.distributed.barrier()

        toc = time.time()

        if not args.distributed or (args.distributed and args.rank == 0):
          print('Epoch Time(s): {:.4f}'.format(toc - tic))

          if epoch >= 5:
            avg += toc - tic
          if epoch % args.eval_every == 0: #and epoch != 0:
            if args.task == "classification":
              train_acc = evaluate(model, g, g.ndata['features'], labels, train_mask, datapins, device, args)
              model.train()
              print('Train Acc {:.4f}'.format(train_acc))
            else:
              tns_pred, tp, fp, tn, fn, acc = evaluate(model, g, g.ndata['features'], slacks, train_mask, datapins, device, args)
              slack_gap = (abs(tns_act[0]) - abs(tns_pred[0]))/abs(tns_act[0])
              model.train()
              with open('train_{}_{}'.format(args.num_hidden,args.num_layers), 'a') as f:
                print('tns_pred: {:.4f}, tns_act: {:.4f} tp: {:d}, fp {:d}, tn: {:d}, fn {:d}, train_acc: {:.4f}, slack_gap: {:.4f}'.format(
                    tns_pred[0], tns_act[0], tp, fp, tn, fn, acc, slack_gap), file=f)


          if epoch % 50 == 0 and epoch != 0:
            if args.distributed:
              save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': model.module.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'features'  : g.ndata['features']
              }, args.checkpoint_file)
            else:
              save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'features'  : g.ndata['features']
              }, args.checkpoint_file)


    if args.distributed:
      th.distributed.barrier()
      if args.rank == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    else:
      print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("GraphSAGE for EDA training")
    argparser.add_argument('--gpu', type=int, default=-1, help="GPU device ID. Use 0 for GPU training")
    argparser.add_argument("--path", type=str, help="directory containing EDA graph and data")
    argparser.add_argument("--graph", type=str, help="EDA Graph")
    argparser.add_argument("--graph-data", type=str, help="EDA Graph Data")
    argparser.add_argument("--all-train", action='store_false', help="use all data for training (default=True)")
    argparser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
    argparser.add_argument('--task', type=str, default='regression')
    argparser.add_argument("--epsilon", type=float, default=0.5, help="training convergence limit")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=4096)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='4,4')
    argparser.add_argument('--eval-fan-out', type=str, default='100,100')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0, help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--neg-samples', type=int, default=None)
    argparser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
    argparser.add_argument('--checkpoint-file', default='checkpoint', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
    argparser.add_argument('--feat-checkpt-file', default='feat_checkpt', type=str, metavar='PATH',
                            help='path to latest feature checkpoint (default: none)')

    argparser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
    argparser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
    argparser.add_argument('--dist-url', default='tcp://127.0.0.1:12356', type=str,
                            help='url used to set up distributed training')
    argparser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
    args = argparser.parse_args()
    
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("PMI_SIZE", -1))
        if args.world_size == -1: args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 
    data = EDADataset(args.path, args.graph, args.graph_data, args.resume, args.all_train, nets_as_edges=False)
    df = np.load('{}/datapins.npz'.format(args.path))
    datapins = df['datapins']
    dfs = np.load('{}/datapin_slacks.npz'.format(args.path))
    datapin_slacks = dfs['slacks']
    train_mask = data.train_mask
    if not args.all_train:
      test_mask = data.test_mask
    labels = th.LongTensor(data.cell_labels)
    slacks = th.FloatTensor(data.cell_slacks)
    n_classes = data.num_cell_labels
    n_slacks = data.num_cell_slacks
    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    data.graph.clear()

    # Pack data

    if args.resume == '':
      cp_path = os.environ.get('CHECKPOINT_PATH', '')
      cp_post = '.pth.tar'
      assert cp_path != ''
      dtnow = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
      args.checkpoint_file = cp_path + '/' + args.checkpoint_file + '_' + str(args.num_hidden) + '_' + str(args.num_layers) + '_' + dtnow + cp_post
      args.feat_checkpt_file = cp_path + 'features_' + str(args.num_hidden) + '_' + str(args.num_layers) + '_' + dtnow
      g.ndata['features'] = th.cat((th.Tensor(data.cell_features),th.Tensor(data.net_features)))
      in_feats = g.ndata['features'].shape[1]
    else:
      args.checkpoint_file = args.resume

    if args.task == 'classification':
      if args.resume == '':
        data = train_mask, test_mask, in_feats, labels, n_classes, datapins, datapin_slacks, g
      else:
        data = train_mask, test_mask, labels, n_classes, datapins, datapin_slacks, g
    else:
      if args.resume == '':
        data = train_mask, in_feats, slacks, n_slacks, datapins, datapin_slacks, g
      else:
        data = train_mask, slacks, n_slacks, datapins, datapin_slacks, g


    run(args, device, data)
