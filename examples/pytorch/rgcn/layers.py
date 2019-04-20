import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import pdb
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.decomposition import tucker
from tensorly.random import check_random_state
from tensorly.tucker_tensor import tucker_to_tensor
from torch_scatter import scatter_add

random_state = 1234
rng = check_random_state(random_state)

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if self.bias == True:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)

        g.ndata['h'] = node_repr

class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                                    self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight
        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index) * edges.data['norm']}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                #pdb.set_trace()
                #score = torch.sigmoid(torch.bmm(msg.unsqueeze(1), msg.unsqueeze(2)))
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)


class RGCNBasisEmbeddingLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisEmbeddingLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.is_input_layer = is_input_layer 
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        if self.is_input_layer:
            self.embed = nn.Embedding(self.in_feat, self.out_feat)
            self.embed.weight.data.requires_grad = False
            nn.init.xavier_normal_(self.embed.weight, gain=nn.init.calculate_gain('sigmoid'))
        else:
            # add basis weights
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                    self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                        self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))


    def propagate(self, g):
        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                #embed = weight.view(-1, self.out_feat)
                #index = edges.data['type'] * self.in_feat + edges.src['id']
                #pdb.set_trace()
                msg = self.embed(edges.src['id'])
                return {'msg': msg * edges.data['norm']}
        else:
            if self.num_bases < self.num_rels:
                # generate all weights from bases
                weight = self.weight.view(self.num_bases,
                                          self.in_feat * self.out_feat)
                weight = torch.matmul(self.w_comp, weight).view(
                                        self.num_rels, self.in_feat, self.out_feat)
            else:
                weight = self.weight


            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                #pdb.set_trace()
                #score = torch.sigmoid(torch.bmm(msg.unsqueeze(1), msg.unsqueeze(2)))
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)





class RGCNTuckerLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False, rank=3, input_dropout=0.2):
        super(RGCNTuckerLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.num_bases = self.num_rels
        self.rank = rank

        self.ranks = [self.rank, self.rank, self.rank]
        # add basis weights
        weight = torch.empty((self.num_bases, self.in_feat,
                                                self.out_feat))

        self.core = nn.Parameter(torch.empty((self.rank, self.rank, self.rank)))
        self.factor_1 = nn.Parameter(torch.empty((weight.shape[0], self.ranks[0])))
        self.factor_2 = nn.Parameter(torch.empty((weight.shape[1], self.ranks[1])))
        self.factor_3 = nn.Parameter(torch.empty((weight.shape[2], self.ranks[2])))

        nn.init.xavier_normal_(self.core, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.orthogonal_(self.factor_1, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.orthogonal_(self.factor_2, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.orthogonal_(self.factor_3, gain=nn.init.calculate_gain('sigmoid'))

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.bnw = torch.nn.BatchNorm1d(self.in_feat)

        # self.factors = nn.ParameterList([])
        # for f_i,f in enumerate(factors):
        #     fac = nn.Parameter(f)
        #     # self.register_parameter('tucker_factor_{}'.format(f_i), fac)
        #     self.factors.append(fac)
        # # self.weight_full = nn.Parameter(self.weight.torch())
        # cores = []
        # for c_i, core in enumerate(self.weight.cores):
        #     core = nn.Parameter(core)
        #     self.register_parameter('tucker_core_{}'.format(c_i), core)
        #     cores.append(core)
        # self.weight.cores = cores
        #
        # Us = []
        # for u_i, u in enumerate(self.weight.Us):
        #     u = nn.Parameter(u)
        #     self.register_parameter('tucker_Us_{}'.format(u_i), u)
        #     Us.append(u)
        #
        # self.weight.Us = Us
        # self.model_params = nn.ParameterList(cores + Us)


    # def parameters(self, recurse=True):
    #     for c in self.weight.cores:
    #         if c.requires_grad:
    #             yield c
    #     for U in self.weight.Us:
    #         if U is not None and U.requires_grad:
    #             yield U
    
    def propagate(self, g):
        # pdb.set_trace()
        core_rank = self.core.shape[0]
        cr = list(self.core.shape)
        core = self.core
        for fi in range(3):
            f = getattr(self, "factor_{}".format(fi+1))
            cr.append(f.shape[0])
            core = torch.matmul(f, core.view(core_rank, -1)).view(f.shape[0], cr[1], -1)
            core = core.permute(1, 2, 0).contiguous()
            cr.pop(0)
        weight = self.bnw(self.input_dropout(core))
        # no float?
        # weight = self.weight.float()
        # weight = tucker_to_tensor(self.core, [self.factor_1, self.factor_2, self.factor_3])

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                #weight = test_tucker_to_tensor(self.core, [self.factor_1, self.factor_2, self.factor_3]).float()
                #weight = self.weight
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index) * edges.data['norm']}
        else:
            def msg_func(edges):
                #weight = tucker_to_tensor(self.core, [self.factor_1, self.factor_2, self.factor_3]).float()
                #weight = self.weight
                w = weight.index_select(0, edges.data['type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                # pdb.set_trace()
                # score = torch.sigmoid(torch.bmm(msg.unsqueeze(1), msg.unsqueeze(2)))
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)


def test_tucker_to_tensor(core, factors):
    core_rank = core.shape[0]
    cr = list(core.shape)
    for f in factors:
        cr.append(f.shape[0])
        core = torch.matmul(f, core.view(core_rank, -1)).view(f.shape[0], cr[1], -1)
        core = core.permute(1,2,0).contiguous()
        cr.pop(0)
    return core



class RGCNBasisAttentionLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None, attention_heads=1,
                 activation=None, is_input_layer=False, self_loop=False, dropout=0.0):
        super(RGCNBasisAttentionLayer, self).__init__(in_feat, out_feat, bias, activation,  self_loop=self_loop,
                                             dropout=dropout)
        self.is_input_layer = is_input_layer
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.attention_heads = attention_heads
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        if self.is_input_layer:
            self.embed = nn.Embedding(self.in_feat, self.out_feat)
            self.embed.weight.data.requires_grad = False
            nn.init.xavier_normal_(self.embed.weight, gain=nn.init.calculate_gain('sigmoid'))
        else:
            # add basis weights
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
            self.a_weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.attention_heads))

            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
                self.w_comp_a = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.a_weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.w_comp_a,
                                        gain=nn.init.calculate_gain('relu'))


    def propagate(self, g):
        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                #embed = weight.view(-1, self.out_feat)
                #index = edges.data['type'] * self.in_feat + edges.src['id']
                #pdb.set_trace()
                msg = self.embed(edges.src['id'])
                return {'msg': msg * edges.data['norm']}
        else:
            if self.num_bases < self.num_rels:
                # generate all weights from bases
                weight = self.weight.view(self.num_bases,
                                          self.in_feat * self.out_feat)
                weight = torch.matmul(self.w_comp, weight).view(
                                        self.num_rels, self.in_feat, self.out_feat)
                a_weight = self.a_weight.view(self.num_bases,
                                          self.in_feat * self.attention_heads)
                a_weight = torch.matmul(self.w_comp_a, a_weight).view(
                                        self.num_rels, self.in_feat, self.attention_heads)
            else:
                weight = self.weight
                a_weight = self.a_weight

            def msg_func(edges):
                #pdb.set_trace()
                w = weight.index_select(0, edges.data['type'])
                w_a = a_weight.index_select(0, edges.data['type'])
                atten = torch.tanh(torch.bmm(edges.src['h'].unsqueeze(1), w_a)).squeeze()
                atten = torch.exp(atten)
                indexes = edges.src['id']
                atten_sum = scatter_add(atten, indexes)
                atten_sum = torch.gather(atten_sum, 0, indexes)
                atten = atten.div(atten_sum)
                # atten_soft = torch.zeros_like(atten).to(atten.device)
                # # apply grouped softmax
                # for node_i in range(edges.src['id'].max().item()):
                #     indexes = (edges.src['id'] == node_i).nonzero()
                #     atten_soft[indexes] = F.softmax(atten[indexes],dim=0)
                #pdb.set_trace()
                w = w * atten.unsqueeze(1).unsqueeze(1)
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze() 
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)




class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

