#!/usr/bin/env/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch as th
import torch.nn as nn
def init_weight(weight, method):
    """
    Initialize parameters
    Args:
        weight: a Parameter object
        method: initialization method
    """
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')


class CentroidDistance(nn.Module):

    def __init__(self,dim,num_centroid, manifold,c):
        super(CentroidDistance, self).__init__()

        self.manifold = manifold
        self.c = c
        # centroid embedding
        self.num_centroid = num_centroid
        self.dim = dim
        self.centroid_embedding = nn.Embedding(
            num_centroid, dim,
            sparse=False,
            scale_grad_by_freq=False,
        )
        self.manifold.init_embed(self.centroid_embedding.weight,self.c)

    def forward(self, node_repr, mask):
        """
        Args:
            node_repr: [node_num, embed_size]
            mask: [node_num, 1] 1 denote real node, 0 padded node
        return:
            graph_centroid_dist: [1, num_centroid]
            node_centroid_dist: [1, node_num, num_centroid]
        """
        node_num = node_repr.size(0)

        # broadcast and reshape node_repr to [node_num * num_centroid, embed_size]
        node_repr = node_repr.unsqueeze(1).expand(
                                                -1,
                                                self.num_centroid,
                                                -1).contiguous().view(-1, self.dim)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, embed_size]

        centroid_repr = self.centroid_embedding.weight
        # centroid_repr = self.manifold.proj(centroid_repr,self.c)
        # centroid_repr = self.manifold.expmap0(centroid_repr,self.c)


        centroid_repr = centroid_repr.unsqueeze(0).expand(
                                                node_num,
                                                -1,
                                                -1).contiguous().view(-1, self.dim)
        # get distance
        node_centroid_dist = torch.sqrt(self.manifold.sqdist(node_repr, centroid_repr,self.c))
        node_centroid_dist = node_centroid_dist.view(1, node_num, self.num_centroid) * mask
        # average pooling over nodes
        graph_centroid_dist = th.sum(node_centroid_dist, dim=1) / th.sum(mask)
        return graph_centroid_dist, node_centroid_dist
