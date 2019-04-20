"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.contrib.data import load_data
import dgl.function as fn
from functools import partial

from layers import RGCNBasisLayer,RGCNBasisEmbeddingLayer, RGCNBasisAttentionLayer, RGCNTuckerLayer
from model import BaseRGCN
import matplotlib.pyplot as plt

class EntityClassify(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RGCNBasisLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self, idx):
        return RGCNBasisLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNBasisLayer(self.h_dim, self.out_dim, self.num_rels,self.num_bases,
                         activation=partial(F.softmax, dim=1))


class EntityClassifyAttention(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RGCNBasisAttentionLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self, idx):
        return RGCNBasisAttentionLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNBasisAttentionLayer(self.h_dim, self.out_dim, self.num_rels,self.num_bases,
                         activation=partial(F.softmax, dim=1))

class EntityClassifyTucker(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RGCNTuckerLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True,rank=self.core_t, input_dropout=args.tucker_dropout)

    def build_hidden_layer(self, idx):
        return RGCNTuckerLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, rank=self.core_t, input_dropout=args.tucker_dropout)

    def build_output_layer(self):
        return RGCNTuckerLayer(self.h_dim, self.out_dim, self.num_rels,self.num_bases,
                         activation=partial(F.softmax, dim=1), rank=self.core_t, input_dropout=args.tucker_dropout)

class EntityClassifyEmbedding(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RGCNBasisEmbeddingLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self, idx):
        return RGCNBasisEmbeddingLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNBasisEmbeddingLayer(self.h_dim, self.out_dim, self.num_rels,self.num_bases,
                         activation=partial(F.softmax, dim=1))


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        print(n)
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    print(ave_grads)
    print(max_grads)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.savefig('rgcn_grads.png')

def main(args):
    # load graph data
    data = load_data(args.dataset, bfs_level=args.bfs_level, relabel=args.relabel)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes
    labels = data.labels
    train_idx = data.train_idx
    test_idx = data.test_idx

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # edge type and normalization factor
    edge_type = torch.from_numpy(data.edge_type)
    edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)
    labels = torch.from_numpy(labels).view(-1)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        labels = labels.cuda()
        device = 'cuda'
    else:
        device = 'cpu'

    # create graph
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(data.edge_src, data.edge_dst)
    g.edata.update({'type': edge_type, 'norm': edge_norm})

    # create model
    if args.attention:
        print("Using Attention")
        model = EntityClassifyAttention(len(g),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_cuda=use_cuda)
    elif args.tucker:
        print("Using Tucker decomposition")
        model = EntityClassifyTucker(len(g),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           core_t=args.core_t,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_cuda=use_cuda)
    elif args.embedding:
        print("Using Node Embedding Lookup")
        model = EntityClassifyEmbedding(len(g),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_cuda=use_cuda) 
    else:
        model = EntityClassify(len(g),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_cuda=use_cuda)

    if use_cuda:
        model.cuda()

    # print number of params
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def print_parameters(model):
        for n,p in model.named_parameters():
            print(n,p.numel())
    
    gparams = count_parameters(model)
    print("Params : ", gparams)
    # optimizer
    # import pdb; pdb.set_trace()
    for name, par in model.named_parameters():
        print(name, par.shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model.forward(g)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        if args.tucker and args.orthogonal_reg:
            # apply orthogonal regularixation to the factor matrices
            reg = 1e-6
            orth_loss = torch.empty((1), requires_grad=True, device=device, dtype=torch.float32)
            for name, param in model.named_parameters():
                if 'factor' in name:
                    param_flat = param.view(param.shape[0], -1)
                    sym = torch.mm(param_flat, torch.t(param_flat))
                    sym -= torch.eye(param_flat.shape[0], device=device)
                    orth_loss = orth_loss + (reg * sym.sum())
            loss += orth_loss.item()
        t1 = time.time()
        loss.backward()
        #import pdb; pdb.set_trace()
        #plot_grad_flow(model.named_parameters())
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
    print()

    model.eval()
    logits = model.forward(g)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("[togrep] | Run: {} | Test Accuracy: {:.4f} | Test loss: {:.4f} | Params: {}".format(args.run, test_acc, test_loss.item(), gparams))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("-a","--attention", default=False, action='store_true',
            help="add attention") 
    parser.add_argument("-t","--tucker", default=False, action='store_true',
            help="use tucker decomp")
    parser.add_argument("--tucker_dropout", type=float, default=0.2,
                        help="tucker weight dropout probability")
    parser.add_argument("--orthogonal_reg", action='store_true',
                        help="apply orthogonal regularization to factor matrices")
    parser.add_argument("-c", "--core-t", type=int, default=5,
            help="core tensor representation") 
    parser.add_argument("-em","--embedding", default=False, action='store_true',
            help="use embedding")
    parser.add_argument("-r", "--run", type=int, default=0,
                        help="run number")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)

