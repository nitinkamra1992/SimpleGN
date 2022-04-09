from itertools import chain, repeat, permutations

import pdb
import torch
import torch.nn as nn

from torch_scatter import scatter_mean

############## Graph class and Graph operators ##############


class Graph:
    """ Basic graph class.

    Notations:
        N: Total number of nodes.
        E: Total number of edges.
        B: Total number of batches/connected components (for a single unbatched
            connected graph, this should always be 1).
        d_n: The dimension of node attributes.
        d_e: The dimension of edge attributes.
        d_g: The dimension of global attributes.

    The class stores the graph structure in the following Pytorch tensors:
        node_attr: Node attributes of shape (N, d_n) such that
            node_attr[i,:] are the node attributes of the i-th node.
        edge_attr: Edge attributes of shape (E, d_e) such that
            edge_attr[j,:] are the edge attributes of the j-th edge.
        global_attr: Global attributes of shape (B, d_g) where B is typically
            the batch size but can in general be the number of disconnected
            components in the graph. global_attr[b,:] is the global attribute
            of the b-th batch element/connected component.
        edge_index: Tensor of long type and shape (2, E) such that
            s,r = edge_index[:,j] -> j-th edge connects the nodes s and r.
        ng_index: Node to global index of long type and shape (N,) such that
            b = ng_index[i] -> i-th node belongs to b-th batch element.
        eg_index: Edge to global index of long type and shape (E,) such that
            b = eg_index[j] -> j-th edge belongs to b-th batch element.
    """

    def __init__(self, node_attr, edge_attr=None, global_attr=None,
                 edge_index=None, ng_index=None, eg_index=None):
        """ Initializes a new graph.
        """
        assert node_attr is not None, "node_attr must be provided"

        self.node_attr = node_attr
        self.edge_attr = edge_attr
        self.global_attr = global_attr
        self.edge_index = edge_index
        self.ng_index = ng_index
        self.eg_index = eg_index

    def num_nodes(self):
        if self.node_attr is not None:
            return self.node_attr.shape[0]
        else:
            return 0

    def num_edges(self):
        if self.edge_attr is not None:
            return self.edge_attr.shape[0]
        elif self.edge_index is not None:
            return self.edge_index.shape[1]
        else:
            return 0

    def num_batches(self):
        if self.global_attr is not None:
            return self.global_attr.shape[0]
        elif self.ng_index is not None:
            return int(self.ng_index.data.max()) + 1
        elif self.num_nodes() > 0:
            return 1
        else:
            return 0

    def soft_copy(self):
        """ This function creates a soft copy of the input graph i.e. it does not
            copy the internal tensors but rather just references them.
        """
        return Graph(self.node_attr, self.edge_attr, self.global_attr,
                     self.edge_index, self.ng_index, self.eg_index)


def decompose_graph(graph):
    return (graph.node_attr, graph.edge_attr, graph.global_attr,
            graph.edge_index, graph.ng_index, graph.eg_index)


############## Generate graphs from node data ##############


def get_fc_edge_indices(sizes, device):
    """ Get fully connected edge_index for nodes in a batch with the i-th
        element of batch having sizes[i] nodes.
    Args:
        sizes: List with N items with each item being the #nodes in each
            batch element
    Returns:
        edge_index: torch.tensor of shape (2, E) with all directed edges
            all nodes belonging to the same batch element (no self-loops)
    """
    L = [[(s,t) for s, t in permutations(range(sum(sizes[:i]), sum(sizes[:i])+sizes[i]), 2)] for i in range(len(sizes))]
    edge_index = torch.tensor(list(chain.from_iterable(L)), device=device)
    edge_index = edge_index.transpose(0, 1) if edge_index.dim() == 2 else torch.zeros(2, 0, dtype=torch.long, device=device)
    return edge_index


def get_fc_ng_indices(sizes, device):
    """ Get fully connected ng_index for nodes in a batch with the i-th
        element of batch having sizes[i] nodes.
    Args:
        sizes: List with B items with each item being the #nodes in each
            batch element
    Returns:
        ng_index: torch.tensor of shape (N,) giving membership of each node
            to a batch
    """
    ng_index = torch.tensor(list(chain.from_iterable([repeat(i, sizes[i]) for i in range(len(sizes))])), device=device)
    return ng_index


def get_fc_eg_indices(sizes, device):
    """ Get fully connected eg_index for nodes in a batch with the i-th
        element of batch having sizes[i] nodes.
    Args:
        sizes: List with B items with each item being the #nodes in each
            batch element
    Returns:
        eg_index: torch.tensor of shape (E,) giving membership of each edge
            to a batch
    """
    eg_index = torch.tensor(list(chain.from_iterable([repeat(i, sizes[i] * (sizes[i]-1)) for i in range(len(sizes))])), device=device)
    return eg_index


def nodes_to_graph(nodes, sizes, edge_attr_dim, global_attr_dim):
    """ Converts a node attribute tensor of shape (N,D) having N nodes with
        features of size D each (potentially batched) into a batched graph.
    Args:
        nodes: torch.tensor of shape (N,D)
        sizes: List with B items with each item being the #nodes in each
            batch element
        edge_attr_dim: Edge attribute dimension
        global_attr_dim: Global attribute dimension
    Returns:
        graph: Batched graph of type Graph with each component fully connected
    """
    N, D = nodes.shape
    device = nodes.device

    _node_attr = nodes
    _edge_index = get_fc_edge_indices(sizes, device)
    _ng_index = get_fc_ng_indices(sizes, device)
    _eg_index = get_fc_eg_indices(sizes, device)

    B = len(sizes)
    E = _edge_index.shape[1]
    
    _edge_attr = torch.zeros((E, edge_attr_dim), device=device)
    _global_attr = torch.zeros((B, global_attr_dim), device=device)
    
    graph = Graph(_node_attr, _edge_attr, _global_attr,
                  _edge_index, _ng_index, _eg_index)
    return graph


############## Default neural nets ##############


def get_default_net(in_dim, out_dim, latent_dim=32):
    return nn.Sequential(nn.Linear(in_dim, latent_dim),
                         nn.ReLU(),
                         nn.Linear(latent_dim, out_dim)
                    )


############## Graph Network Blocks ##############


class GlobalBlock(nn.Module):
    """Global block, f_g.
    A block that updates the global features of each graph based on
    the previous global features, the aggregated features of the
    edges of the graph, and the aggregated features of the nodes of the graph.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 latent_dim=32,
                 use_edges=True,
                 use_nodes=True,
                 use_globals=True,
                 edge_reducer=scatter_mean,
                 node_reducer=scatter_mean,
                 custom_func=None):
        
        super(GlobalBlock, self).__init__()
        
        if not (use_nodes or use_edges or use_globals):
            raise ValueError("At least one of use_edges, use_nodes or use_globals must be True.")
    
        self._use_edges = use_edges    # not need to differentiate sent/received edges.
        self._use_nodes = use_nodes
        self._use_globals = use_globals
        self._edge_reducer = edge_reducer
        self._node_reducer = node_reducer
        
        # f_g is a function R^in_dim -> R^out_dim
        if custom_func:
            # Customized function can be used for self.net instead of default function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.net = get_default_net(in_dim, out_dim, latent_dim)
    
    def forward(self, graph, node_mask=None, edge_mask=None):
        # Decompose graph
        node_attr, edge_attr, global_attr, edge_index, ng_index, eg_index = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        num_edges = graph.num_edges()
        num_nodes = graph.num_nodes()
        num_batches = graph.num_batches()
        
        globals_to_collect = []
        
        if self._use_globals:
            globals_to_collect.append(global_attr)    # global_attr.shape=(B, d_g)
            
        if self._use_edges:
            if num_edges > 0:
                # not need to differentiate sent/received edges.
                try:
                    if edge_mask is None:
                        agg_edges = self._edge_reducer(edge_attr, eg_index, dim=0, dim_size=num_batches)
                    else:
                        edge_mask_temp = edge_mask.unsqueeze(1).repeat(1, edge_attr.shape[1])
                        agg_edges = self._edge_reducer(edge_attr * edge_mask_temp, eg_index, dim=0, dim_size=num_batches)
                except:
                    raise ValueError("reducer should be one of scatter_* [add, mul, max, min, mean]")
            else:
                agg_edges = torch.zeros((num_batches, edge_attr.shape[1]), device=edge_attr.device)
            globals_to_collect.append(agg_edges)
        
        if self._use_nodes:
            try:
                if node_mask is None:
                    agg_nodes = self._node_reducer(node_attr, ng_index, dim=0, dim_size=num_batches)
                else:
                    node_mask_temp = node_mask.unsqueeze(1).repeat(1, node_attr.shape[1])
                    agg_nodes = self._node_reducer(node_attr * node_mask_temp, ng_index, dim=0, dim_size=num_batches)
            except:
                raise ValueError("reducer should be one of scatter_* [add, mul, max, min, mean]")
            globals_to_collect.append(agg_nodes)
        
        collected_globals = torch.cat(globals_to_collect, dim=-1)
        graph.global_attr = self.net(collected_globals)    # Update
        
        return graph



class EdgeBlock(nn.Module):
    """Edge block, f_e.
    Update the features of each edge based on the previous edge features,
    the features of the adjacent nodes, and the global features.
    """
    
    def __init__(self,
                 in_dim,
                 out_dim,
                 latent_dim=32,
                 use_edges=True,
                 use_sender_nodes=True,
                 use_receiver_nodes=True,
                 use_globals=True,
                 custom_func=None):
        
        super(EdgeBlock, self).__init__()
        
        if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
            raise ValueError("At least one of use_edges, use_sender_nodes, "
                             "use_receiver_nodes or use_globals must be True.")
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._use_edges = use_edges
        self._use_sender_nodes = use_sender_nodes
        self._use_receiver_nodes = use_receiver_nodes
        self._use_globals = use_globals
    
        # f_e() is a function: R^in_dim -> R^out_dim
        # Customized functions can be used for self.nets instead of default function.
        # It is highly recommended to use nn.Sequential() types.
        if custom_func is None:
            custom_func = get_default_net(in_dim, out_dim, latent_dim)
        self.net = custom_func


    def forward(self, graph, node_mask=None, edge_mask=None):
        # Decompose graph
        node_attr, edge_attr, global_attr, edge_index, ng_index, eg_index = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        num_edges = graph.num_edges()

        if num_edges > 0:
            edges_to_collect = []

            if self._use_edges:
                edges_to_collect.append(edge_attr)
                
            if self._use_sender_nodes:
                senders_attr = node_attr[senders_idx, :]
                if node_mask is None:
                    edges_to_collect.append(senders_attr)
                else:
                    node_mask_sender = node_mask.unsqueeze(1).repeat(1, node_attr.shape[1])[senders_idx, :]
                    edges_to_collect.append(senders_attr * node_mask_sender)
                
            if self._use_receiver_nodes:
                receivers_attr = node_attr[receivers_idx, :]
                if node_mask is None:
                    edges_to_collect.append(receivers_attr)
                else:
                    node_mask_receiver = node_mask.unsqueeze(1).repeat(1, node_attr.shape[1])[receivers_idx, :]
                    edges_to_collect.append(receivers_attr * node_mask_receiver)
            
            if self._use_globals:
                expanded_global_attr = global_attr[eg_index, :]
                edges_to_collect.append(expanded_global_attr)
                
            collected_edges = torch.cat(edges_to_collect, dim=-1)

            # Update edge_attr
            if edge_mask is None:
                graph.edge_attr = self.net(collected_edges)
            else:
                edge_mask_temp = edge_mask.unsqueeze(1).repeat(1, self.out_dim)
                graph.edge_attr = edge_mask_temp * self.net(collected_edges)

        else:
            graph.edge_attr = torch.zeros((0, self.out_dim), device=edge_attr.device)

        return graph



class NodeBlock(nn.Module):
    """Node block, f_v.
    Update the features of each node based on the previous node features,
    the aggregated features of the received edges,
    the aggregated features of the sent edges, and the global features.
    """
    
    def __init__(self,
                 in_dim,
                 out_dim,
                 latent_dim=32,
                 use_nodes=True,
                 use_sent_edges=False,
                 use_received_edges=True,
                 use_globals=True,
                 sent_edges_reducer=scatter_mean,
                 received_edges_reducer=scatter_mean,
                 custom_func=None):
        
        super(NodeBlock, self).__init__()

        if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
            raise ValueError("At least one of use_received_edges, use_sent_edges, "
                             "use_nodes or use_globals must be True.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self._use_nodes = use_nodes
        self._use_sent_edges = use_sent_edges
        self._use_received_edges = use_received_edges
        self._use_globals = use_globals
        self._sent_edges_reducer = sent_edges_reducer
        self._received_edges_reducer = received_edges_reducer

        # f_v() is a function: R^in_dim -> R^out_dim
        # Customized functions can be used for self.nets instead of default function.
        # It is highly recommended to use nn.Sequential() types.
        if custom_func is None:
            custom_func = get_default_net(in_dim, out_dim, latent_dim)
        self.net = custom_func

    def forward(self, graph, node_mask=None, edge_mask=None):
        # Decompose graph
        node_attr, edge_attr, global_attr, edge_index, ng_index, eg_index = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        num_edges = graph.num_edges()
        num_nodes = graph.num_nodes()
        num_batches = graph.num_batches()
        
        nodes_to_collect = []
        
        if self._use_nodes:
            nodes_to_collect.append(node_attr)
            
        if self._use_sent_edges:
            if num_edges > 0:
                try:
                    if edge_mask is None:
                        agg_sent_edges = self._sent_edges_reducer(edge_attr, senders_idx, dim=0, dim_size=num_nodes)
                    else:
                        edge_mask_temp = edge_mask.unsqueeze(1).repeat(1, edge_attr.shape[1])
                        agg_sent_edges = self._sent_edges_reducer(edge_attr * edge_mask_temp, senders_idx, dim=0, dim_size=num_nodes)
                except:
                    raise ValueError("reducer should be one of scatter_* [add, mul, max, min, mean]")
            else:
                agg_sent_edges = torch.zeros((num_nodes, edge_attr.shape[1]), device=edge_attr.device)
            nodes_to_collect.append(agg_sent_edges)
            
        if self._use_received_edges:
            if num_edges > 0:
                try:
                    if edge_mask is None:
                        agg_received_edges = self._received_edges_reducer(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)
                    else:
                        edge_mask_temp = edge_mask.unsqueeze(1).repeat(1, edge_attr.shape[1])
                        agg_received_edges = self._received_edges_reducer(edge_attr * edge_mask_temp, receivers_idx, dim=0, dim_size=num_nodes)
                except:
                    raise ValueError("reducer should be one of scatter_* [add, mul, max, min, mean]")
            else:
                agg_received_edges = torch.zeros((num_nodes, edge_attr.shape[1]), device=edge_attr.device)
            nodes_to_collect.append(agg_received_edges)

        if self._use_globals:
            expanded_global_attr = global_attr[ng_index, :]
            nodes_to_collect.append(expanded_global_attr)

        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        # Update node_attr
        if node_mask is None:
            graph.node_attr = self.net(collected_nodes)
        else:
            node_mask_temp = node_mask.unsqueeze(1).repeat(1, self.out_dim)
            graph.node_attr = node_mask_temp * self.net(collected_nodes)

        return graph


        
class NodeBlockInd(NodeBlock):
    """Node-level feature transformation.
    Each node is considered independently. (No edge is considered.)
    
    Args:
        in_dim: input dimension of node representations.
        out_dim: output dimension of node representations.
            (node embedding size)
            
    (N, d_v) -> (N, out_dim)
    NodeBlockInd(graph) -> updated graph
    """
    
    def __init__(self, in_dim, out_dim, latent_dim=32, custom_func=None):
        super(NodeBlockInd, self).__init__(in_dim, out_dim, latent_dim,
                                           use_nodes=True,
                                           use_sent_edges=False,
                                           use_received_edges=False,
                                           use_globals=False,
                                           sent_edges_reducer=None,
                                           received_edges_reducer=None,
                                           custom_func=custom_func)



class EdgeBlockInd(EdgeBlock):
    """Edge-level feature transformation.
    Each edge is considered independently. (No node is considered.)
    
    Args:
        in_dim: input dimension of edge representations.
        out_dim: output dimension of edge representations.
            (edge embedding size)
    
    (E, d_e) -> (E, out_dim)
    EdgeBlockInd(graph) -> updated graph
    """

    def __init__(self, in_dim, out_dim, latent_dim=32, custom_func=None):
        super(EdgeBlockInd, self).__init__(in_dim, out_dim, latent_dim,
                                           use_edges=True,
                                           use_sender_nodes=False,
                                           use_receiver_nodes=False,
                                           use_globals=False,
                                           custom_func=custom_func)

    

class GlobalBlockInd(GlobalBlock):
    """Global-level feature transformation.
    No edge/node is considered.
    
    Args:
        in_dim: input dimension of global representations.
        out_dim: output dimension of global representations.
            (global embedding size)
    
    (1, d_g) -> (1, out_dim)
    GlobalBlockInd(graph) -> updated graph
    """
    
    def __init__(self, in_dim, out_dim, latent_dim=32, custom_func=None):
        
        super(GlobalBlockInd, self).__init__(in_dim, out_dim, latent_dim,
                                             use_edges=False,
                                             use_nodes=False,
                                             use_globals=True,
                                             edge_reducer=None,
                                             node_reducer=None,
                                             custom_func=custom_func)



class GNConv(nn.Module):
    """Graph Networks (https://arxiv.org/abs/1806.01261) module.
    (This code is mainly based on 
      https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py)
      
    A graph network takes a graph as input and returns a graph as output.
    The input graph has edge-level, node-level and global-level (u) attributes.
    The output graph has the same structure but updated attributes.
    
    Notations: 
        h represents attribute
        h_i, h_j: i-th and j-th node attributes, respectively.
        h_ij: Edge attributes connecting i and j node. (If directed, h_ij is i->j edge)
        u: Global attributes
        AGG(...): Aggregated attributes. (It is usually aggregated edge or node attributes.)
    
    Args:
        edge_model_block: f_e(h_ij, h_i, h_j, u), per-edge computations. Use nn.Module()
        node_model_block: f_v(h_i, AGG(h_ij), AGG(h_ji), u), per-node computations. Use nn.Module()
        global_model_block: f_g(AGG(all nodes), AGG(all edges), u), global attribute computations. Use nn.Module()
        use_edge_block: Enable using edge block (default: True)
        use_node_block: Enable using node block (default: True)
        use_global_block: Enable using global block (default: True)
    """
    
    def __init__(self,
                 edge_model_block,
                 node_model_block,
                 global_model_block,
                 use_edge_block=True,
                 use_node_block=True,
                 use_global_block=True):
        
        super(GNConv, self).__init__()
        
        # f_e, f_v, f_g
        self.edge_model_block = edge_model_block
        self.node_model_block = node_model_block
        self.global_model_block = global_model_block
        self._use_edge_block = use_edge_block
        self._use_node_block = use_node_block
        self._use_global_block = use_global_block
        
    def reset_parameters(self):
        for m in self.edge_model_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.node_model_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for m in self.global_model_block.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        
    def forward(self, graph, node_mask=None, edge_mask=None):
        """This is a high-level module.
        Read graph and
        1. update edge-level
        2. update node-level
        3. update global-level
        and return the updated graph
        
        Args:
            graph: Graph                
        """
            
        if self._use_edge_block:
            # Edge-level update
            graph = self.edge_model_block(graph, node_mask, edge_mask)    # (E, d_e + d_n + d_n + d_g) -> (E, d_e')
        
        if self._use_node_block:
            # Node-level update
            graph = self.node_model_block(graph, node_mask, edge_mask)    # (N, d_n + d_e' + d_e' + d_g) -> (N, d_n')
        
        if self._use_global_block:
            # Global-level update
            graph = self.global_model_block(graph, node_mask, edge_mask)  # (B, d_n' + d_e' + d_g) -> (B, d_g')

        return graph
