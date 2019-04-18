import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import pdb
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.tucker_tensor import tucker_to_tensor

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
                 activation=None, is_input_layer=False, core_t=5):
        super(RGCNTuckerLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.num_bases = self.num_rels
        self.core_tensor = core_t

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.core_tensor, self.core_tensor,
                                                self.core_tensor))
        self.base_m = nn.Parameter(torch.Tensor(self.num_bases, self.core_tensor))
        self.in_feat_m = nn.Parameter(torch.Tensor(self.in_feat, self.core_tensor))
        self.out_feat_m = nn.Parameter(torch.Tensor(self.out_feat, self.core_tensor))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.base_m, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.in_feat_m, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.out_feat_m, gain=nn.init.calculate_gain('relu'))
    
    def propagate(self, g): 
        #w_ = torch.matmul(self.base_m, self.weight.view(self.core_tensor, -1))
        #w_ = w_.view(self.num_bases, self.core_tensor, -1).permute(1,2,0).contiguous()
        #w_ = torch.matmul(self.in_feat_m, w_.view(self.core_tensor, -1))
        #w_ = w_.view(self.in_feat, self.core_tensor, -1).permute(1,2,0).contiguous()
        #w_ = torch.matmul(self.out_feat_m, w_.view(self.core_tensor, -1))
        #w_ = w_.view(self.out_feat, self.num_bases, -1).permute(1,2,0).contiguous()
        #weight = w_
        weight = tucker_to_tensor(self.weight, [self.base_m, self.in_feat_m, self.out_feat_m])

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
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)





class RGCNBasisAttentionLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisAttentionLayer, self).__init__(in_feat, out_feat, bias, activation)
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
        self.a_weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.in_feat))

        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))
            self.w_comp_a = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu')) 
        nn.init.xavier_uniform_(self.a_weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.w_comp_a,
                                    gain=nn.init.calculate_gain('relu'))


    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                                    self.num_rels, self.in_feat, self.out_feat)
            a_weight = self.a_weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            a_weight = torch.matmul(self.w_comp_a, a_weight).view(
                                    self.num_rels, self.in_feat, self.in_feat)
        else:
            weight = self.weight
            a_weight = self.a_weight

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
                w_a = a_weight.index_select(0, edges.data['type'])
                pdb.set_trace()
                atten = F.sigmoid(torch.bmm(torch.bmm(edges.src['h'].unsqueeze(1), w_a), edges.dst['h'].unsqueeze(2))).squeeze() 
                w = w * atten.unsqueeze(1)
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

