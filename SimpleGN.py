import pdb
import torch
import torch.nn as nn

from torch_scatter import scatter_mean, scatter_max, scatter_add

import sys
sys.path.append('./')
from utils.SimpleGN.utils import nodes_to_graph, GlobalBlock, NodeBlock, EdgeBlock, NodeBlockInd, GNConv


class SimpleGN(nn.Module):
    ''' GN block edge_net -> node_net -> global_net structure

             *----------*    *----------*    *------------*
             |          |    |          |    |            |
        G--->| edge_net |--->| node_net |--->| global_net |--->Q(G,actions)
             |          |    |          |    |            |
             *----------*    *----------*    *------------*
    '''
    def __init__(self, *args, **kwargs):
        super(SimpleGN, self).__init__()

        self.input_dim = kwargs['input_dim']
        self.K = kwargs['K']
        self.n_actions = kwargs['n_actions']
        self.latent_dim = kwargs['latent_dim']

        self.node_attr_dim = kwargs['node_attr_dim']
        self.edge_attr_dim = kwargs['edge_attr_dim']
        self.reducer = {'mean': scatter_mean, 'sum': scatter_add, 'max': scatter_max}[kwargs.get('reducer', 'mean')]

        # Encoder
        self.encoder = NodeBlockInd(self.input_dim, self.node_attr_dim, self.latent_dim)

        # GN
        #### edge_net
        in_dim = self.node_attr_dim*2
        edge_net = EdgeBlock(
            in_dim, self.edge_attr_dim, self.latent_dim,
            use_edges=False, use_sender_nodes=True, use_receiver_nodes=True, use_globals=False
        )
        #### node_net
        in_dim = self.node_attr_dim + self.edge_attr_dim
        node_net = NodeBlock(
            in_dim, self.node_attr_dim, self.latent_dim,
            use_nodes=True, use_sent_edges=False, use_received_edges=True, use_globals=False,
            sent_edges_reducer=self.reducer, received_edges_reducer=self.reducer
        )
        #### global_net
        in_dim = self.node_attr_dim + self.edge_attr_dim
        global_net = GlobalBlock(
            in_dim, self.n_actions, self.latent_dim,
            use_edges=True, use_nodes=True, use_globals=False,
            edge_reducer=self.reducer, node_reducer=self.reducer
        )
        #### GN block
        self.GN = GNConv(
            edge_net, node_net, global_net,
            use_edge_block=True, use_node_block=True, use_global_block=True
        )

    def forward(self, theta):
        # Assuming size of theta: (B*K, input_dim)
        device = theta.device
        B = theta.shape[0] // self.K
        sizes = [self.K for _ in range(B)]
        G = nodes_to_graph(theta, sizes, self.edge_attr_dim, self.n_actions)
        G2 = self.GN(self.encoder(G))
        return G2.global_attr
