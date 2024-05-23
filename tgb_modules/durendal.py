import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score

import random

import bisect

import gc
import copy

from itertools import permutations

import pandas as pd

from torch_geometric.utils import negative_sampling, structured_negative_sampling
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import (
    RandomLinkSplit,
    NormalizeFeatures,
    Constant,
    OneHotDegree,
)
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv, Linear, GCN, GAT

import torch
import networkx as nx
import numpy as np

import copy

"""
Source: DURENDAL: multirelational/traineval.py
URL: https://github.com/manuel-dileo/durendal/blob/main/multirelational/traineval.py
"""


def reverse_insort(a, x, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Function useful to compute MRR.
    """
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > a[mid]:
            hi = mid
        else:
            lo = mid + 1
    a.insert(lo, x)
    return lo


def compute_mrr(real_scores, fake_scores):
    srr = 0
    count = 0
    for i, score in enumerate(real_scores):
        try:
            fake_scores_cp = copy.copy([fake_scores[i]])
        except IndexError:
            break
        rank = reverse_insort(fake_scores_cp, score)
        rr = 1 / (rank + 1)  # index starts from zero
        srr += rr
        count += 1
    return srr / count


def durendal_test(model, i_snap, test_data, data, device="cpu"):

    model.eval()

    test_data = test_data.to(device)
    edge_types = list(data.edge_index_dict.keys())

    h_dict, *_ = model(test_data.x_dict, test_data.edge_index_dict, test_data, i_snap)

    tot_avgpr = 0
    tot_mrr = 0

    num_rel = 0

    for edge_t in edge_types:

        h = h_dict[edge_t]
        pred_cont = torch.sigmoid(h).cpu().detach().numpy()

        num_pos = len(test_data[edge_t].edge_label_index[0]) // 2
        h_fake = h[num_pos:]

        fake_preds = torch.sigmoid(h_fake).cpu().detach().numpy()
        edge_label = test_data[edge_t].edge_label.cpu().detach().numpy()

        if len(edge_label) > 0:
            avgpr_score = average_precision_score(edge_label, pred_cont)
            mrr_score = compute_mrr(pred_cont[:num_pos], fake_preds)

            tot_avgpr += avgpr_score
            tot_mrr += mrr_score
            num_rel += 1

    return tot_avgpr / num_rel, tot_mrr / num_rel


def durendal_train_single_snapshot(
    model,
    data,
    i_snap,
    train_data,
    val_data,
    test_data,
    past_dict_1,
    past_dict_2,
    optimizer,
    device="cpu",
    num_epochs=50,
    verbose=False,
):

    mrr_val_max = 0
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    best_past_dict_1 = {}
    best_past_dict_2 = {}

    tol = 5e-2

    edge_types = list(data.edge_index_dict.keys())

    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()

        pred_dict, past_dict_1, past_dict_2 = model(
            train_data.x_dict,
            train_data.edge_index_dict,
            train_data,
            i_snap,
            past_dict_1,
            past_dict_2,
        )

        preds = torch.Tensor()
        edge_labels = torch.Tensor()
        for edge_t in edge_types:
            preds = torch.cat((preds, pred_dict[edge_t]), -1)
            edge_labels = torch.cat(
                (edge_labels, train_data[edge_t].edge_label.type_as(pred_dict[edge_t])),
                -1,
            )

        # compute loss function based on all edge types
        loss = model.loss(preds, edge_labels)
        loss = torch.autograd.Variable(loss, requires_grad=True)

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = "Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}"
        avgpr_score_val, mrr_val = durendal_test(model, i_snap, val_data, data, device)

        """
        if mrr_val_max-tol < mrr_val:
            mrr_val_max = mrr_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = copy.deepcopy(model)
        else:
            break
        
        #print(f'Epoch: {epoch} done')
            
        """
        gc.collect()
        if avgpr_val_max - tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_past_dict_1 = past_dict_1
            best_past_dict_2 = past_dict_2
            best_model = model
        else:
            break

    avgpr_score_test, mrr_test = durendal_test(model, i_snap, test_data, data, device)

    if verbose:
        print(f"Best Epoch: {best_epoch}")
    # print(f'Best Epoch: {best_epoch}')

    return (
        best_model,
        avgpr_score_test,
        mrr_test,
        best_past_dict_1,
        best_past_dict_2,
        optimizer,
    )


"""
Change the function to only do the training part
"""


def durendal_train_single_snapshot_only_train(
    model,
    data,
    i_snap,
    train_data,
    val_data,
    past_dict_1,
    past_dict_2,
    optimizer,
    device="cpu",
    num_epochs=50,
    verbose=False,
):

    mrr_val_max = 0
    avgpr_val_max = 0
    best_model = model
    train_data = train_data.to(device)
    best_epoch = -1
    best_past_dict_1 = {}
    best_past_dict_2 = {}

    tol = 5e-2

    edge_types = list(data.edge_index_dict.keys())

    for epoch in range(num_epochs):
        model.train()
        ## Note
        ## 1. Zero grad the optimizer
        ## 2. Compute loss and backpropagate
        ## 3. Update the model parameters
        optimizer.zero_grad()

        pred_dict, past_dict_1, past_dict_2 = model(
            train_data.x_dict,
            train_data.edge_index_dict,
            train_data,
            i_snap,
            past_dict_1,
            past_dict_2,
        )

        preds = torch.Tensor()
        edge_labels = torch.Tensor()
        for edge_t in edge_types:
            preds = torch.cat((preds, pred_dict[edge_t]), -1)
            edge_labels = torch.cat(
                (edge_labels, train_data[edge_t].edge_label.type_as(pred_dict[edge_t])),
                -1,
            )

        # compute loss function based on all edge types
        loss = model.loss(preds, edge_labels)
        loss = torch.autograd.Variable(loss, requires_grad=True)

        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        ##########################################

        log = "Epoch: {:03d}\n AVGPR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n MRR Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n F1-Score Train: {:.4f}, Val: {:.4f}, Test: {:.4f}\n Loss: {}"
        avgpr_score_val, mrr_val = durendal_test(model, i_snap, val_data, data, device)

        """
        if mrr_val_max-tol < mrr_val:
            mrr_val_max = mrr_val
            best_epoch = epoch
            best_current_embeddings = current_embeddings
            best_model = copy.deepcopy(model)
        else:
            break
        
        #print(f'Epoch: {epoch} done')
            
        """
        gc.collect()
        if avgpr_val_max - tol <= avgpr_score_val:
            avgpr_val_max = avgpr_score_val
            best_epoch = epoch
            best_past_dict_1 = past_dict_1
            best_past_dict_2 = past_dict_2
            best_model = model
        else:
            break

    if verbose:
        print(f"Best Epoch: {best_epoch}")
    # print(f'Best Epoch: {best_epoch}')

    return best_model, best_past_dict_1, best_past_dict_2, optimizer


"""
Modified the training mode to only train the model
"""


def training_durendal_uta(snapshots, hidden_conv_1, hidden_conv_2, device="cpu"):
    num_snap = len(snapshots)
    hetdata = copy.deepcopy(snapshots[0])
    edge_types = list(hetdata.edge_index_dict.keys())

    lr = 0.001
    weight_decay = 5e-3

    in_channels = {node: len(v[0]) for node, v in hetdata.x_dict.items()}
    num_nodes = {node: len(v) for node, v in hetdata.x_dict.items()}

    # DURENDAL
    durendal = RDurendal(
        in_channels,
        num_nodes,
        hetdata.metadata(),
        hidden_conv_1=hidden_conv_1,
        hidden_conv_2=hidden_conv_2,
    )

    durendal.reset_parameters()

    durendalopt = torch.optim.Adam(
        params=durendal.parameters(), lr=lr, weight_decay=weight_decay
    )

    past_dict_1 = {}
    for node in hetdata.x_dict.keys():
        past_dict_1[node] = {}
    for src, r, dst in hetdata.edge_index_dict.keys():
        past_dict_1[src][r] = torch.Tensor(
            [[0 for j in range(hidden_conv_1)] for i in range(hetdata[src].num_nodes)]
        )
        past_dict_1[dst][r] = torch.Tensor(
            [[0 for j in range(hidden_conv_1)] for i in range(hetdata[dst].num_nodes)]
        )

    past_dict_2 = {}
    for node in hetdata.x_dict.keys():
        past_dict_2[node] = {}
    for src, r, dst in hetdata.edge_index_dict.keys():
        past_dict_2[src][r] = torch.Tensor(
            [[0 for j in range(hidden_conv_2)] for i in range(hetdata[src].num_nodes)]
        )
        past_dict_2[dst][r] = torch.Tensor(
            [[0 for j in range(hidden_conv_2)] for i in range(hetdata[dst].num_nodes)]
        )

    for i in range(num_snap - 1):
        # CREATE TRAIN + VAL + TEST SET FOR THE CURRENT SNAP
        snapshot = copy.deepcopy(snapshots[i])

        link_split = RandomLinkSplit(num_val=0.0, num_test=0.20, edge_types=edge_types)

        het_train_data, _, het_val_data = link_split(snapshot)

        # het_test_data = copy.deepcopy(snapshots[i+1])
        # future_link_split = RandomLinkSplit(num_val=0, num_test=0, edge_types = edge_types) #useful only for negative sampling
        # het_test_data, _, _ = future_link_split(het_test_data)

        # TRAIN AND TEST THE MODEL FOR THE CURRENT SNAP
        durendal, past_dict_1, past_dict_2, durendalopt = (
            durendal_train_single_snapshot_only_train(
                durendal,
                snapshot,
                i,
                het_train_data,
                het_val_data,
                past_dict_1,
                past_dict_2,
                durendalopt,
            )
        )

        # SAVE AND DISPLAY EVALUATION
        print(f"Snapshot: {i} completed\n")

        gc.collect()

    print("DURENDAL Training Completed")

    return durendal, past_dict_1, past_dict_2, durendalopt


"""
Source: DURENDAL: src/durendal.py
URL: https://github.com/manuel-dileo/durendal/blob/main/src/durendal.py
"""

import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, GRUCell
from torch_geometric.data import Data
import random
import gc
import copy
from itertools import permutations

from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATv2Conv,
    GINConv,
    Linear,
    GraphConv,
    GATConv,
)

import networkx as nx
import numpy as np

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax

import math

from sklearn.metrics import *


class DurendalConv(MessagePassing):
    """
    A class that perform the message-passing operation according to the DURENDAL architecture.
    In DurendalConv, messages are exchanged between each edge type and
    partial node representations for each relation type are computed for each node type.
    """

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr=None, node_dim=0, **kwargs)
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.dropout = dropout

        # self.proj = nn.ModuleDict()
        # for node_type, in_channels in self.in_channels.items():
        # self.proj[node_type] = Linear(in_channels, out_channels)

        # A message-passing layer for each relation type
        self.conv = nn.ModuleDict()
        for edge_type in metadata[1]:
            src, _, dst = edge_type
            edge_type = "__".join(edge_type)
            self.conv[edge_type] = GraphConv(
                (in_channels[src], in_channels[dst]), out_channels
            )

        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.proj)
        reset(self.conv)

    def forward(
        self, x_dict: Dict[NodeType, Tensor], edge_index_dict: Dict[EdgeType, Adj]
    ):
        x_node_dict, out_dict = {}, {}

        # Iterate over node types to linear project the node features in the same space:
        for node_type, x in x_dict.items():
            # x_node_dict[node_type] = self.proj[node_type](x)
            x_node_dict[node_type] = x_dict[node_type]
            out_dict[node_type] = {}

        # Iterate over edge types to perform convolution:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, r_type, dst_type = edge_type
            edge_type = "__".join(edge_type)
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            x = (x_src, x_dst)
            out = self.conv[edge_type](x, edge_index)

            out = F.relu(out)
            out_dict[dst_type][r_type] = out

        # Retrieve the node representations even if they have no in-edges (filtered out by default by PyG)
        for node_type, out in out_dict.items():
            if out == {}:
                for edge_type in edge_index_dict.keys():
                    src_type, r_type, _ = edge_type
                    if src_type == node_type:
                        out_dict[node_type][r_type] = x_node_dict[node_type]

        return out_dict


class DurendalGATConv(MessagePassing):
    """
    A class that perform the message-passing operation according to the DURENDAL architecture.
    In DurendalConv, messages are exchanged between each edge type and
    partial node representations for each relation type are computed for each node type.
    """

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr=None, node_dim=0, **kwargs)
        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metadata = metadata
        self.dropout = dropout

        # self.proj = nn.ModuleDict()
        # for node_type, in_channels in self.in_channels.items():
        # self.proj[node_type] = Linear(in_channels, out_channels)

        # A message-passing layer for each relation type
        self.conv = nn.ModuleDict()
        for edge_type in metadata[1]:
            src, _, dst = edge_type
            edge_type = "__".join(edge_type)
            self.conv[edge_type] = GATv2Conv(
                (in_channels[src], in_channels[dst]), out_channels
            )

        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.proj)
        reset(self.conv)

    def forward(
        self, x_dict: Dict[NodeType, Tensor], edge_index_dict: Dict[EdgeType, Adj]
    ):
        x_node_dict, out_dict = {}, {}

        # Iterate over node types to linear project the node features in the same space:
        for node_type, x in x_dict.items():
            # x_node_dict[node_type] = self.proj[node_type](x)
            x_node_dict[node_type] = x_dict[node_type]
            out_dict[node_type] = {}

        # Iterate over edge types to perform convolution:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, r_type, dst_type = edge_type
            edge_type = "__".join(edge_type)
            x_src = x_node_dict[src_type]
            x_dst = x_node_dict[dst_type]
            x = (x_src, x_dst)
            out = self.conv[edge_type](x, edge_index)

            out = F.relu(out)
            out_dict[dst_type][r_type] = out

        # Retrieve the node representations even if they have no in-edges (filtered out by default by PyG)
        for node_type, out in out_dict.items():
            if out == {}:
                for edge_type in edge_index_dict.keys():
                    src_type, r_type, _ = edge_type
                    if src_type == node_type:
                        out_dict[node_type][r_type] = x_node_dict[node_type]

        return out_dict


# Below, update modules for Update-Then-Aggregate schema:


class SemanticUpdateGRU(torch.nn.Module):
    """
    Update the partial representation of nodes using GRU Cell.
    """

    def __init__(
        self,
        n_channels,
    ):
        super(SemanticUpdateGRU, self).__init__()
        self.updater = GRUCell(n_channels, n_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.updater.reset_parameters()

    def forward(self, current_in_dict, past_in_dict=None):
        if past_in_dict is None:
            return current_in_dict
        out_dict = {}
        # For each node type, for each relation type, update the node states.
        for node_type, current_in in current_in_dict.items():
            out_dict[node_type] = {}
            for r_type, current_emb in current_in.items():
                past_emb = past_in_dict[node_type][r_type]
                out = torch.Tensor(
                    self.updater(current_emb.clone(), past_emb.clone()).detach().numpy()
                )
                out_dict[node_type][r_type] = out
        return out_dict


class SemanticUpdateMLP(torch.nn.Module):
    """
    Update the partial representation of nodes using ConcatMLP.
    """

    def __init__(
        self,
        n_channels,
    ):
        super(SemanticUpdateMLP, self).__init__()
        self.updater = Linear(n_channels * 2, n_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.updater.reset_parameters()

    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            out_dict[node_type] = {}
            for r_type, current_emb in current_in.items():
                past_emb = past_in_dict[node_type][r_type]
                hin = torch.cat((current_emb.clone(), past_emb.clone()), dim=1)
                out = torch.Tensor(self.updater(hin).detach().numpy())
                out_dict[node_type][r_type] = out
        return out_dict


class SemanticUpdateWA(torch.nn.Module):
    """
    Update the partial representation of nodes using a weighted average.
    """

    def __init__(self, n_channels, tau=0.20):  # weight to past information
        super(SemanticUpdateWA, self).__init__()
        self.tau = tau

    def reset_parameters(self):
        pass

    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            out_dict[node_type] = {}
            for r_type, current_emb in current_in.items():
                past_emb = past_in_dict[node_type][r_type]
                out = torch.Tensor(
                    (self.tau * past_emb.clone() + (1 - self.tau) * current_emb.clone())
                    .detach()
                    .numpy()
                )
                out_dict[node_type][r_type] = out
        return out_dict


# Below, update modules for aggregate-then-update schema


class HetNodeUpdateMLP(torch.nn.Module):
    """
    Implementation of a temporal update embedding module for heterogeneous nodes using ConcatMLP.
    (aggregate-then-update paradigm)
    """

    def __init__(self, in_channels, metadata):
        super(HetNodeUpdateMLP, self).__init__()

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.metadata = metadata

        self.update = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.update[node_type] = Linear(in_channels * 2, in_channels)

    def reset_parameters(self):
        for layer in self.update.values():
            layer.reset_parameters()

    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            past_in = past_in_dict[node_type]
            update_in = torch.cat((current_in.clone(), past_in.clone()), dim=1)
            out_dict[node_type] = torch.Tensor(
                self.update[node_type](update_in).detach().numpy()
            )
        return out_dict


class HetNodeUpdateGRU(torch.nn.Module):
    """
    Implementation of a temporal update embedding module for heterogeneous nodes using ConcatMLP.
    (aggregate-then-update paradigm)
    """

    def __init__(self, in_channels, metadata):
        super(HetNodeUpdateGRU, self).__init__()

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.metadata = metadata

        self.update = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.update[node_type] = GRUCell(in_channels, in_channels)

    def reset_parameters(self):
        for layer in self.update.values():
            layer.reset_parameters()

    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            past_in = past_in_dict[node_type]
            out_dict[node_type] = torch.Tensor(
                self.update[node_type](current_in, past_in).detach().numpy()
            )
        return out_dict


class HetNodeUpdateTA(MessagePassing):
    """
    Implementation of a temporal update embedding module for heterogeneous nodes using temporal attention (DyHan [Yang et al., 2020])
    (aggregate-then-update paradigm)
    Specifically, DyHan utilizes Scaled Dot-Product Attention
    """

    def __init__(self, in_channels, metadata, **kwargs):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.metadata = metadata

        self.k_lin = Linear(in_channels, in_channels, bias=False)
        self.q_lin = Linear(in_channels, in_channels, bias=False)
        self.v_lin = Linear(in_channels, in_channels, bias=False)

    def reset_parameters(self):
        self.k_lin.reset_parameters()
        self.q_lin.reset_parameters()
        self.v_lin.reset_parameters()

    def scaled_dot_product_attention(
        self, query, key, value, attn_mask=None, is_causal=False, scale=None
    ) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        query = None
        key = None
        attn_weight += attn_bias
        attn_bias = None
        gc.collect()
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def group_by_temporal_attention(self, xs: List[Tensor]):
        out = xs[-1]
        Q = self.q_lin(out)
        K = self.k_lin(out)
        V = self.v_lin(out)
        return self.scaled_dot_product_attention(Q, K, V)

    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            xs = []
            past_in = past_in_dict[node_type]
            xs.append(past_in)
            xs.append(current_in)
            out_dict[node_type] = self.group_by_temporal_attention(xs)
            xs = None
            gc.collect()
        return out_dict


class HetNodeUpdatePE(torch.nn.Module):
    """
    Implementation of a temporal update embedding module using positional encoding.
    Temporal update embedding module as defined by HTGNN (Fan et al, 2021)
    (aggregate-then-update paradigm)
    """

    def __init__(self, in_channels, metadata):
        super(HetNodeUpdatePE, self).__init__()

        self.in_channels = in_channels
        self.metadata = metadata

        self.q_lin = Linear(in_channels, in_channels)
        self.k_lin = Linear(in_channels, in_channels)
        self.v_lin = Linear(in_channels, in_channels)

    def reset_parameters(self):
        self.q_lin.reset_parameters()
        self.k_lin.reset_parameters()
        self.v_lin.reset_parameters()

    def positional_encoding(self, x, t):
        n = x.size(1)
        indices = torch.arange(n)
        encoding = torch.where(
            indices % 2 == 0,
            torch.sin(t / (1000 ** (2 * indices / n))),
            torch.cos(t / (1000 ** (2 * indices / n))),
        )
        result = x + encoding
        return result

    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            past_in = past_in_dict[node_type]
            current_pe = self.positional_encoding(current_in, 1)
            past_pe = self.positional_encoding(past_in, 0)
            q = self.q_lin(current_pe)
            k = self.k_lin(past_pe)
            gamma = q * k
            v = self.v_lin(current_pe)
            out_dict[node_type] = F.leaky_relu(gamma * v)
        return out_dict


class HetNodeUpdateGate(MessagePassing):
    """
    Implementation of a temporal update embedding module for heterogeneous nodes
    using the temporal gate of RE-GCN (https://dl.acm.org/doi/10.1145/3404835.3462963)
    (aggregate-then-update paradigm)
    """

    def __init__(self, in_channels, metadata, **kwargs):
        super(HetNodeUpdateGate, self).__init__()

        self.in_channels = in_channels
        self.metadata = metadata

        self.u_lin = Linear(in_channels, in_channels)

    def reset_parameters(self):
        self.u_lin.reset_parameters()

    def temporal_gate(self, current, past):
        u = torch.sigmoid(self.u_lin(past).sum(dim=1)).unsqueeze(dim=1)
        out = (u * current) + ((1 - u) * past)
        return out

    def forward(self, current_in_dict, past_in_dict):
        out_dict = {}
        for node_type, current_in in current_in_dict.items():
            past_in = past_in_dict[node_type]
            out_dict[node_type] = self.temporal_gate(current_in, past_in)
            gc.collect()
        return out_dict


class HetNodeUpdateFake(MessagePassing):
    """
    Implementation of a fake temporal update embedding module that
    do nothing and just forward the current embeddings in output.
    """

    def __init__(self, in_channels, metadata, **kwargs):
        super(HetNodeUpdateFake, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, current_in_dict, past_in_dict):
        return current_in_dict


# Below, aggregation functions at semantic level:
class SemanticAttention(MessagePassing):
    """
    Aggregation scheme for partial node representations using semantic-level attention mechanism,
    as described in "Heterogeneous Graph Attention Network" (Wang et al., 2020)
    """

    def __init__(
        self,
        n_channels: int,
        v2: bool = False,
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)
        self.k_lin = nn.Linear(n_channels, n_channels)
        self.q = nn.Parameter(torch.Tensor(1, n_channels))
        self.v2 = v2
        self.reset_parameters()

    def reset_parameters(self):
        self.k_lin.reset_parameters()
        glorot(self.q)

    def group_by_semantic_attention(
        self,
        xs: List[Tensor],
        q: nn.Parameter,
        k_lin: nn.Module,
    ):
        if len(xs) == 0:
            return None
        else:
            num_edge_types = len(xs)
            out = torch.stack(xs)
            if out.numel() == 0:
                return out.view(0, out.size(-1)), None
            if self.v2:
                attn_score = (q * (torch.tanh(k_lin(out)).mean(1))).sum(
                    -1
                )  # see HTGNN by Fan et al.
            else:
                attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(
                    -1
                )  # see the Heterogeneous Graph Attention Network by Wang et al.
            attn = F.softmax(attn_score, dim=0)
            out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out

    def forward(self, in_dict):
        out_dict = {}
        for node_type, partial_in in in_dict.items():
            xs = []
            for _, v in partial_in.items():
                xs.append(v)
            out = self.group_by_semantic_attention(xs, self.q, self.k_lin)
            out_dict[node_type] = out
        return out_dict


"""
Source: DURENDAL: multirelational/multirelational.py
URL: https://github.com/manuel-dileo/durendal/blob/main/multirelational/multirelational.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import GDELT, ICEWS18
from torch.nn import RReLU, Flatten

from torch_geometric_temporal.nn.recurrent import GConvGRU, EvolveGCNH

import random

import gc
import copy

from itertools import permutations

import pandas as pd

import torch_geometric.transforms as T

import networkx as nx
import numpy as np

import json

import sys

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GATv2Conv,
    Linear,
    HANConv,
    HeteroConv,
    SAGEConv,
    HGTConv,
)
from torch_geometric.nn import ComplEx
from torch.nn import GRUCell, Conv1d

from torch_geometric.nn.inits import glorot


def triple_dot(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return (x * y * z).sum(dim=-1)


class RDurendal(torch.nn.Module):
    """
    Durendal update-then-aggregate for multirelational link prediction task.
    """

    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(RDurendal, self).__init__()
        self.conv1 = DurendalConv(in_channels, hidden_conv_1, metadata)
        # self.update1 = SemanticUpdateWA(n_channels=hidden_conv_1, tau=0.1)
        self.update1 = SemanticUpdateGRU(n_channels=hidden_conv_1)
        # self.update1 = SemanticUpdateMLP(n_channels=hidden_conv_1)
        self.agg1 = SemanticAttention(n_channels=hidden_conv_1)

        self.conv2 = DurendalConv(hidden_conv_1, hidden_conv_2, metadata)
        # self.update2 = SemanticUpdateWA(n_channels=hidden_conv_2, tau=0.1)
        self.update2 = SemanticUpdateGRU(n_channels=hidden_conv_2)
        # self.update2 = SemanticUpdateMLP(n_channels=hidden_conv_2)
        self.agg2 = SemanticAttention(n_channels=hidden_conv_2)

        self.post = Linear(hidden_conv_2, 2)

        self.past_out_dict_1 = None
        self.past_out_dict_2 = None

        self.loss_fn = BCEWithLogitsLoss()

        self.metadata = metadata
        self.rel_emb = torch.nn.Parameter(
            torch.randn(len(metadata[1]), 2), requires_grad=True
        )
        self.rel_to_index = {metapath: i for i, metapath in enumerate(metadata[1])}

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.update1.reset_parameters()
        self.agg1.reset_parameters()
        self.conv2.reset_parameters()
        self.update2.reset_parameters()
        self.agg2.reset_parameters()
        self.post.reset_parameters()

    def forward(
        self,
        x_dict,
        edge_index_dict,
        data,
        snap,
        past_out_dict_1=None,
        past_out_dict_2=None,
    ):
        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()

        out_dict = self.conv1(x_dict, edge_index_dict)
        if snap == 0:
            current_dict_1 = out_dict.copy()
        else:
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()
        out_dict = self.agg1(current_dict_1)

        out_dict = self.conv2(out_dict, edge_index_dict)
        if snap == 0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict = self.agg2(current_dict_2)
        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)

        # ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)

            head = out_dict[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            head_re_a = head[:, 0]
            head_im_a = head[:, 1]

            tail = out_dict[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            tail_re_a = tail[:, 0]
            tail_im_a = tail[:, 1]

            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]

            # ComplEx score
            h = torch.Tensor(
                [
                    triple_dot(head_re, rel_re, tail_re)
                    + triple_dot(head_im, rel_re, tail_im)
                    + triple_dot(head_re, rel_im, tail_im)
                    - triple_dot(head_im, rel_im, tail_re)
                    for head_re, head_im, tail_re, tail_im in zip(
                        head_re_a, head_im_a, tail_re_a, tail_im_a
                    )
                ]
            )

            h_dict[edge_t] = h

        return h_dict, current_dict_1, current_dict_2

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


class RATU(torch.nn.Module):
    """
    Durendal aggregate-then-update for multirelational link prediction task.
    """

    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(RATU, self).__init__()
        self.conv1 = DurendalConv(in_channels, hidden_conv_1, metadata)
        self.agg1 = SemanticAttention(n_channels=hidden_conv_1)
        self.update1 = HetNodeUpdateGRU(hidden_conv_1, metadata)

        self.conv2 = DurendalConv(hidden_conv_1, hidden_conv_2, metadata)
        self.agg2 = SemanticAttention(n_channels=hidden_conv_2)
        self.update2 = HetNodeUpdateGRU(hidden_conv_2, metadata)

        self.post = Linear(hidden_conv_2, 2)

        self.past_out_dict_1 = None
        self.past_out_dict_2 = None

        self.loss_fn = BCEWithLogitsLoss()

        self.metadata = metadata
        self.rel_emb = torch.nn.Parameter(
            torch.randn(len(metadata[1]), 2), requires_grad=True
        )
        self.rel_to_index = {metapath: i for i, metapath in enumerate(metadata[1])}

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.update1.reset_parameters()
        self.agg1.reset_parameters()
        self.conv2.reset_parameters()
        self.update2.reset_parameters()
        self.agg2.reset_parameters()
        self.post.reset_parameters()

    def forward(
        self,
        x_dict,
        edge_index_dict,
        data,
        snap,
        past_out_dict_1=None,
        past_out_dict_2=None,
    ):

        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()

        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = self.agg1(out_dict)
        if snap == 0:
            current_dict_1 = out_dict.copy()
        else:
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()

        out_dict = self.conv2(current_dict_1, edge_index_dict)
        out_dict = self.agg2(out_dict)
        if snap == 0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict = current_dict_2.copy()

        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)

        # ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)

            head = out_dict[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            head_re_a = head[:, 0]
            head_im_a = head[:, 1]

            tail = out_dict[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            tail_re_a = tail[:, 0]
            tail_im_a = tail[:, 1]

            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]

            # ComplEx score
            h = torch.Tensor(
                [
                    triple_dot(head_re, rel_re, tail_re)
                    + triple_dot(head_im, rel_re, tail_im)
                    + triple_dot(head_re, rel_im, tail_im)
                    - triple_dot(head_im, rel_im, tail_re)
                    for head_re, head_im, tail_re, tail_im in zip(
                        head_re_a, head_im_a, tail_re_a, tail_im_a
                    )
                ]
            )

            h_dict[edge_t] = h

        return h_dict, current_dict_1, current_dict_2

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


# Below, Temporal Heterogeneous GNN mapped in the DURENDAL framework
class DyHAN(torch.nn.Module):
    """
    DyHAN model (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7148053/).
    DyHAN utilizies edge-level and semantic-level attention -> HANConv
    Then, DyHAN leverages temporal-attention to combine node embeddings over time
    It can be reconducted to a special instance of our framework,
    following aggregate-then-update schema:
        GNN Encoder: GAT
        Semantic Aggregation: Semantic Attention (HAN)
        Embedding update: Temporal Attention
    """

    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(DyHAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_conv_1, metadata)
        self.update1 = HetNodeUpdateTA(hidden_conv_1, metadata)

        self.conv2 = HANConv(hidden_conv_1, hidden_conv_2, metadata)
        self.update2 = HetNodeUpdateTA(hidden_conv_2, metadata)

        self.post = Linear(hidden_conv_2, 2)

        self.past_out_dict_1 = None
        self.past_out_dict_2 = None

        self.loss_fn = BCEWithLogitsLoss()

        self.metadata = metadata

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.update1.reset_parameters()
        self.update2.reset_parameters()
        self.post.reset_parameters()

    def forward(
        self,
        x_dict,
        edge_index_dict,
        data,
        snap,
        past_out_dict_1=None,
        past_out_dict_2=None,
    ):

        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()

        out_dict = self.conv1(x_dict, edge_index_dict)
        if snap == 0:
            current_dict_1 = out_dict.copy()
        else:
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()

        out_dict = self.conv2(current_dict_1, edge_index_dict)
        if snap == 0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()

        out_dict = None

        # DotProduct followed by LogisticRegression as decoder (following the original paper)
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index

            head = current_dict_2[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            tail = current_dict_2[edge_t[2]][edge_label_index[1]]  # embedding dst nodes

            dot_product = torch.mul(head, tail)
            h = self.post(dot_product)
            h = torch.sigmoid(torch.sum(h.clone(), dim=-1))
            h_dict[edge_t] = h

        return h_dict, current_dict_1, current_dict_2

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


class HTGNN(torch.nn.Module):
    """
    HTGNN model (https://dl.acm.org/doi/pdf/10.1145/3583780.3614909).
    HTGNN utilizies edge-level attention -> GAT
                    semantic-level attention -> HAN with attention coefficient following GATv2
                    positional encoding for temporal node embedding
    It can be reconducted to a special instance of our framework,
    following aggregate-then-update schema:
        GNN Encoder: GAT
        Semantic Aggregation: Semantic Attention (From GATv2)
        Embedding update: PE (PositionalEncoding) + Temporal Aggregation
    """

    def __init__(
        self,
        in_channels,
        in_channels_int,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
    ):
        super(HTGNN, self).__init__()
        self.conv1 = DurendalGATConv(
            in_channels, hidden_conv_1, metadata
        )  # GAT for intra-relation (see original paper)
        self.agg1 = SemanticAttention(
            n_channels=hidden_conv_1, v2=True
        )  # Semantic attention for inter-relation
        self.update1 = HetNodeUpdatePE(
            hidden_conv_1, metadata
        )  # PE followed by temporal attention
        self.lin1 = Linear(
            in_channels_int, hidden_conv_1
        )  # Linear layer for weighted skip-connection
        self.delta1 = torch.nn.Parameter(
            torch.Tensor([1])
        )  # weight for skip-connection

        self.conv2 = DurendalGATConv(hidden_conv_1, hidden_conv_2, metadata)
        self.agg2 = SemanticAttention(n_channels=hidden_conv_2, v2=True)
        self.update2 = HetNodeUpdatePE(hidden_conv_2, metadata)
        self.lin2 = Linear(hidden_conv_1, hidden_conv_2)
        self.delta2 = torch.nn.Parameter(torch.Tensor([1]))

        self.post = Linear(hidden_conv_2 * 2, 2)

        self.past_out_dict_1 = None
        self.past_out_dict_2 = None

        self.loss_fn = BCEWithLogitsLoss()

        self.metadata = metadata

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.update1.reset_parameters()
        self.update2.reset_parameters()
        self.agg1.reset_parameters()
        self.agg2.reset_parameters()
        self.post.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(
        self,
        x_dict,
        edge_index_dict,
        data,
        snap,
        past_out_dict_1=None,
        past_out_dict_2=None,
    ):

        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()

        x_dict_lin = {node: self.lin1(x) for node, x in x_dict.items()}
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = self.agg1(out_dict)
        if snap == 0:
            current_dict_1 = out_dict.copy()
        else:
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        current_dict_1 = {
            node: (
                self.delta1 * current_dict_1[node]
                + (1 - self.delta1) * x_dict_lin[node]
            )
            for node in x_dict_lin
        }
        self.past_out_dict_1 = current_dict_1.copy()

        x_dict_lin = {node: self.lin2(x) for node, x in current_dict_1.items()}
        out_dict = self.conv2(current_dict_1, edge_index_dict)
        out_dict = self.agg2(out_dict)
        if snap == 0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        current_dict_2 = {
            node: (
                self.delta2 * current_dict_2[node]
                + (1 - self.delta2) * x_dict_lin[node]
            )
            for node in x_dict_lin
        }
        self.past_out_dict_2 = current_dict_2.copy()

        out_dict = None

        # ConcatMLP as decoder (following the original paper)
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index

            head = current_dict_2[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            tail = current_dict_2[edge_t[2]][edge_label_index[1]]  # embedding dst nodes

            concat = torch.cat((head, tail), dim=1)
            h = self.post(concat)
            h_dict[edge_t] = torch.sum(h, dim=-1)

        return h_dict, current_dict_1, current_dict_2

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


# Below, Temporal Heterogeneous GNN mapped in the DURENDAL framework
class REGCN(torch.nn.Module):
    """
    RE-GCN model (https://dl.acm.org/doi/10.1145/3404835.3462963).
    RE-GCN utilizies a relational GCN as node-encoder -> RGCN
    Then, it leverages a time-gate to combine node embeddings over time.
    Furthermore, it utilizies representations for relations as
        1a) The mean pooling of the node embeddings involved in the relation
        1b) Random inizialized learnable parameters
        1) concatenation between 1a and 1b
        2) Update over time using a GRU unit
    It can be reconducted to a special instance of our framework,
    following aggregate-then-update schema:
        GNN Encoder: GCN
        Semantic Aggregation: R-GCN aggregation
        Embedding update: Time-Gate / GRU
    """

    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata: Metadata,
        hidden_conv_1,
        hidden_conv_2,
        output_conv,
    ):
        super(REGCN, self).__init__()
        # RGCN using HeteroConv with SAGEConv + mean aggregation
        # As suggested in: https://github.com/pyg-team/pytorch_geometric/discussions/3479
        self.conv1 = HeteroConv(
            {
                edge_t: SAGEConv(
                    (in_channels, in_channels), hidden_conv_1, add_self_loops=False
                )
                for edge_t in metadata[1]
            },
            aggr="mean",
        )
        self.update1 = HetNodeUpdateGate(hidden_conv_1, metadata)

        self.conv2 = HeteroConv(
            {
                edge_t: SAGEConv(
                    (hidden_conv_1, hidden_conv_1), hidden_conv_2, add_self_loops=False
                )
                for edge_t in metadata[1]
            },
            aggr="mean",
        )
        self.update2 = HetNodeUpdateGate(hidden_conv_2, metadata)

        self.rel_emb = torch.nn.Parameter(torch.randn(len(metadata[1]), hidden_conv_2))
        self.rel_to_index = {rel: i for i, rel in enumerate(metadata[1])}
        self.update_rel = GRUCell(hidden_conv_2 * 2, hidden_conv_2)

        self.act = RReLU()
        self.flat = Flatten()

        self.linr = Linear(hidden_conv_2 * 2, hidden_conv_2)

        self.conv_h = Conv1d(
            in_channels=hidden_conv_2, out_channels=output_conv, kernel_size=1
        )
        self.conv_r = Conv1d(
            in_channels=hidden_conv_2, out_channels=output_conv, kernel_size=1
        )
        self.conv_t = Conv1d(
            in_channels=hidden_conv_2, out_channels=output_conv, kernel_size=1
        )

        self.output_conv = output_conv
        self.post = Linear(output_conv * 3, 2)

        self.past_out_dict_1 = None
        self.past_out_dict_2 = None
        self.past_R = None

        self.loss_fn = BCEWithLogitsLoss()

        self.metadata = metadata

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.update1.reset_parameters()
        self.update2.reset_parameters()
        self.linr.reset_parameters()
        self.conv_h.reset_parameters()
        self.conv_r.reset_parameters()
        self.conv_t.reset_parameters()
        self.post.reset_parameters()

    def forward(
        self,
        x_dict,
        edge_index_dict,
        data,
        snap,
        past_out_dict_1=None,
        past_out_dict_2=None,
        past_R=None,
    ):

        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()
        if past_R is not None:
            self.past_R = past_R.clone()

        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: self.act(out) for node, out in out_dict.items()}
        if snap == 0:
            current_dict_1 = out_dict.copy()
        else:
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()

        out_dict = self.conv2(current_dict_1, edge_index_dict)
        out_dict = {node: self.act(out) for node, out in out_dict.items()}
        if snap == 0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict = None

        count_types = 0
        for edge_type in self.metadata[1]:
            rel_emb = self.rel_emb[self.rel_to_index[edge_type]].unsqueeze(0)
            dst_type = edge_type[2]
            avg_node = torch.mean(
                current_dict_2[dst_type][edge_index_dict[edge_type][1]], dim=0
            ).unsqueeze(0)
            r = torch.cat((avg_node, rel_emb), dim=1)
            if count_types == 0:
                R = r.clone()
                count_types += 1
            else:
                R = torch.cat((R, r))
        if snap == 0:
            current_R = self.linr(R).clone()
        else:
            current_R = torch.Tensor(self.update_rel(R, self.past_R).detach().numpy())
        self.past_R = current_R.clone()

        # ConvTransE as decoder (following the coriginal paper)
        h_dict = dict()
        flat = self.flat
        out_conv = self.output_conv
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            if edge_label_index[0].size(0) == 0:
                h_dict[edge_t] = torch.Tensor([])
                continue
            head = current_dict_2[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            tail = current_dict_2[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            rel = current_R[self.rel_to_index[edge_t]].unsqueeze(0)
            # CONV
            head_conv = self.conv_h(head.reshape(head.size(1), head.size(0))).reshape(
                head.size(0), out_conv
            )
            tail_conv = self.conv_h(tail.reshape(tail.size(1), tail.size(0))).reshape(
                tail.size(0), out_conv
            )
            rel_conv = (
                self.conv_r(rel.reshape(rel.size(1), 1))
                .reshape(rel.size(0), out_conv)
                .repeat(head.size(0), 1)
            )

            concat = torch.cat(
                (flat(head_conv), flat(rel_conv), flat(tail_conv)), dim=1
            )
            h = self.post(concat)
            h_dict[edge_t] = torch.sum(h, dim=-1)

        return h_dict, current_dict_1, current_dict_2, current_R

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


# Below, other GNN baselines


class RHAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_conv_1, hidden_conv_2, metadata):
        super(RHAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_conv_1, metadata)
        self.conv2 = HANConv(hidden_conv_1, hidden_conv_2, metadata)

        self.post = Linear(hidden_conv_2, 2)

        self.loss_fn = BCEWithLogitsLoss()

        self.metadata = metadata
        self.rel_emb = torch.nn.Parameter(
            torch.randn(len(metadata[1]), 2), requires_grad=True
        )
        self.rel_to_index = {metapath: i for i, metapath in enumerate(metadata[1])}

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.post.reset_parameters()

    def forward(self, x_dict, edge_index_dict, data):
        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: out.relu() for node, out in out_dict.items()}
        out_dict = self.conv2(out_dict, edge_index_dict)
        out_dict = {node: out.relu() for node, out in out_dict.items()}

        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)

        # ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)

            head = out_dict[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            head_re_a = head[:, 0]
            head_im_a = head[:, 1]

            tail = out_dict[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            tail_re_a = tail[:, 0]
            tail_im_a = tail[:, 1]

            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]

            # ComplEx score
            h = torch.Tensor(
                [
                    triple_dot(head_re, rel_re, tail_re)
                    + triple_dot(head_im, rel_re, tail_im)
                    + triple_dot(head_re, rel_im, tail_im)
                    - triple_dot(head_im, rel_im, tail_re)
                    for head_re, head_im, tail_re, tail_im in zip(
                        head_re_a, head_im_a, tail_re_a, tail_im_a
                    )
                ]
            )

            h_dict[edge_t] = h

        return h_dict

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


class RHEGCN(torch.nn.Module):
    def __init__(self, in_channels, num_nodes, edge_types):
        super(RHEGCN, self).__init__()

        self.conv1 = HeteroConv(
            {edge_t: EvolveGCNH(num_nodes, in_channels) for edge_t in edge_types}
        )
        self.post = torch.nn.Linear(in_channels, 2)

        self.edge_types = edge_types
        self.rel_emb = torch.nn.Parameter(
            torch.randn(len(edge_types), 2), requires_grad=True
        )
        self.rel_to_index = {metapath: i for i, metapath in enumerate(edge_types)}

        self.loss_fn = BCEWithLogitsLoss()

    def reset_parameters(self):
        self.post.reset_parameters()

    def forward(self, x_dict, edge_index_dict, data):

        out_dict = self.conv1(x_dict, edge_index_dict)
        out_dict = {node: out.relu() for node, out in out_dict.items()}

        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)

        # ComplEx decoder
        h_dict = dict()
        for edge_t in self.edge_types:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)

            head = out_dict[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            head_re_a = head[:, 0]
            head_im_a = head[:, 1]

            tail = out_dict[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            tail_re_a = tail[:, 0]
            tail_im_a = tail[:, 1]

            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]

            # ComplEx score
            h = torch.Tensor(
                [
                    triple_dot(head_re, rel_re, tail_re)
                    + triple_dot(head_im, rel_re, tail_im)
                    + triple_dot(head_re, rel_im, tail_im)
                    - triple_dot(head_im, rel_im, tail_re)
                    for head_re, head_im, tail_re, tail_im in zip(
                        head_re_a, head_im_a, tail_re_a, tail_im_a
                    )
                ]
            )

            h_dict[edge_t] = h

        return h_dict

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


# Below, Factorization-method baselines


class ComplEx(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_nodes,
        metadata,
    ):
        super(ComplEx, self).__init__()

        init_size = 1e-3
        self.embedding_dim = embedding_dim
        self.metadata = metadata
        self.node_emb = torch.nn.Parameter(torch.randn((num_nodes, 2 * embedding_dim)))
        self.rel_emb = torch.nn.Parameter(
            torch.randn((len(metadata[1]), 2 * embedding_dim))
        )
        self.rel_to_index = {metapath: i for i, metapath in enumerate(metadata[1])}

        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, x_dict, edge_index_dict, data):

        out_dict = dict()

        out_dict["node"] = self.node_emb

        # ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)

            head = out_dict[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            head_re_a = head[:, : self.embedding_dim]
            head_im_a = head[:, self.embedding_dim :]

            tail = out_dict[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            tail_re_a = tail[:, : self.embedding_dim]
            tail_im_a = tail[:, self.embedding_dim :]

            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[: self.embedding_dim]
            rel_im = rel[self.embedding_dim :]

            # ComplEx score
            h = torch.Tensor(
                [
                    triple_dot(head_re, rel_re, tail_re)
                    + triple_dot(head_im, rel_re, tail_im)
                    + triple_dot(head_re, rel_im, tail_im)
                    - triple_dot(head_im, rel_im, tail_re)
                    for head_re, head_im, tail_re, tail_im in zip(
                        head_re_a, head_im_a, tail_re_a, tail_im_a
                    )
                ]
            )

            h_dict[edge_t] = h

        return h_dict

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


class TNTComplEx(torch.nn.Module):
    def __init__(self, embedding_dim, num_nodes, metadata, num_timestamps, rnn_size):
        super(TNTComplEx, self).__init__()

        init_size = 1e-3
        self.embedding_dim = embedding_dim
        self.metadata = metadata
        self.node_emb = torch.nn.Parameter(torch.randn((num_nodes, 2 * embedding_dim)))
        self.rel_emb = torch.nn.Parameter(
            torch.randn((len(metadata[1]), 2 * embedding_dim))
        )

        self.rnn_size = rnn_size
        self.num_timestamps = num_timestamps
        self.rnn = torch.nn.GRU(rnn_size, rnn_size)
        self.post_rnn = nn.Linear(rnn_size, 2 * embedding_dim)
        self.h0 = nn.Parameter(torch.randn(1, 1, rnn_size))
        self.rnn_input = nn.Parameter(
            torch.zeros(self.num_timestamps, 1, rnn_size), requires_grad=False
        )

        self.rel_to_index = {metapath: i for i, metapath in enumerate(metadata[1])}

        self.loss_fn = BCEWithLogitsLoss()

    def forward(self, x_dict, edge_index_dict, data, ts):

        out_dict = dict()

        out_dict["node"] = self.node_emb

        time_emb, _ = self.rnn(self.rnn_input, self.h0)
        time_emb = torch.squeeze(time_emb)
        time_emb = self.post_rnn(time_emb)
        time_emb_current = time_emb[ts]
        time_re = time_emb_current[: self.embedding_dim]
        time_im = time_emb_current[self.embedding_dim :]

        # TNTComplEx score
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)

            head = out_dict[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            head_re_a = head[:, : self.embedding_dim]
            head_im_a = head[:, self.embedding_dim :]

            tail = out_dict[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            tail_re_a = tail[:, : self.embedding_dim]
            tail_im_a = tail[:, self.embedding_dim :]

            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[: self.embedding_dim]
            rel_im = rel[self.embedding_dim :]

            # TNTComplEx score
            h = torch.Tensor(
                [
                    torch.sum(
                        (
                            (
                                head_re * rel_re * time_re
                                - head_im * rel_im * time_re
                                - head_im * rel_re * time_im
                                - head_re * rel_im * time_im
                            )
                            * tail_re
                            + (
                                head_im * rel_re * time_re
                                + head_re * rel_im * time_re
                                + head_re * rel_re * time_im
                                - head_im * rel_im * time_im
                            )
                            * tail_im
                        ),
                        -1,
                    )
                    for head_re, head_im, tail_re, tail_im in zip(
                        head_re_a, head_im_a, tail_re_a, tail_im_a
                    )
                ]
            )

            h_dict[edge_t] = h

        return h_dict

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)


# Below, an utility class to repurpose HAN, HGT and R-GCN into Temporal Heterogeneous GNNs.
class HeteroToTemporal(torch.nn.Module):
    """
    Repurposing model for RGCN, HAN and HGT in our framework following the aggregate-then-update scheme
    """

    def __init__(
        self,
        in_channels,
        num_nodes,
        metadata,
        hidden_conv_1,
        hidden_conv_2,
        model="rgcn",
        upd="GRU",
    ):

        super(HeteroToTemporal, self).__init__()

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        edge_types = metadata[1]

        if model == "RGCN":
            # RGCN is realized using GraphConv due to no heterogeneity-aware of GCNConv operator
            # This choice was made according to this github thread https://github.com/pyg-team/pytorch_geometric/discussions/3479
            self.conv1 = HeteroConv(
                {
                    edge_t: GraphConv(
                        (in_channels[edge_t[0]], in_channels[edge_t[2]]),
                        hidden_conv_1,
                        add_self_loops=False,
                    )
                    for edge_t in edge_types
                },
                aggr="sum",
            )
            self.conv2 = HeteroConv(
                {
                    edge_t: GraphConv(
                        (hidden_conv_1, hidden_conv_1),
                        hidden_conv_2,
                        add_self_loops=False,
                    )
                    for edge_t in edge_types
                },
                aggr="sum",
            )
        elif model == "HAN":
            self.conv1 = HANConv(in_channels, hidden_conv_1, metadata)
            self.conv2 = HANConv(hidden_conv_1, hidden_conv_2, metadata)
        elif model == "HGT":
            self.conv1 = HGTConv(in_channels, hidden_conv_1, metadata)
            self.conv2 = HGTConv(hidden_conv_1, hidden_conv_2, metadata)

        if upd == "GRU":
            self.update1 = HetNodeUpdateGRU(hidden_conv_1, metadata)
            self.update2 = HetNodeUpdateGRU(hidden_conv_2, metadata)
        elif upd == "MLP":
            self.update1 = HetNodeUpdateMLP(hidden_conv_1, metadata)
            self.update2 = HetNodeUpdateMLP(hidden_conv_2, metadata)
        elif upd == "PE":
            self.update1 = HetNodeUpdatePE(hidden_conv_1, metadata)
            self.update2 = HetNodeUpdatePE(hidden_conv_2, metadata)
        elif upd == "NO-UPD":
            self.update1 = HetNodeUpdateFake(hidden_conv_1, metadata)
            self.update2 = HetNodeUpdateFake(hidden_conv_1, metadata)

        self.post = Linear(hidden_conv_2, 2)

        # Initialize the loss function to BCEWithLogitsLoss
        self.loss_fn = BCEWithLogitsLoss()

        self.past_out_dict_1 = None
        self.past_out_dict_2 = None

        self.metadata = metadata
        self.rel_emb = torch.nn.Parameter(
            torch.randn(len(metadata[1]), 2), requires_grad=True
        )
        self.rel_to_index = {metapath: i for i, metapath in enumerate(metadata[1])}

        self.upd = upd

        self.reset_parameters()

    def reset_loss(self, loss=BCEWithLogitsLoss):
        self.loss_fn = loss()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.update1.reset_parameters()
        self.update2.reset_parameters()
        self.post.reset_parameters()

    def forward(
        self,
        x_dict,
        edge_index_dict,
        data,
        snap,
        past_out_dict_1=None,
        past_out_dict_2=None,
    ):

        if past_out_dict_1 is not None:
            self.past_out_dict_1 = past_out_dict_1.copy()
        if past_out_dict_2 is not None:
            self.past_out_dict_2 = past_out_dict_2.copy()

        out_dict = self.conv1(x_dict, edge_index_dict)
        if snap == 0:
            current_dict_1 = out_dict.copy()
        else:
            current_dict_1 = out_dict.copy()
            current_dict_1 = self.update1(out_dict, self.past_out_dict_1)
        self.past_out_dict_1 = current_dict_1.copy()

        out_dict = self.conv2(out_dict, edge_index_dict)
        if snap == 0:
            current_dict_2 = out_dict.copy()
        else:
            current_dict_2 = out_dict.copy()
            current_dict_2 = self.update2(out_dict, self.past_out_dict_2)
        self.past_out_dict_2 = current_dict_2.copy()
        out_dict_iter = out_dict.copy()
        for node, h in out_dict_iter.items():
            out_dict[node] = self.post(h)

        # ComplEx decoder
        h_dict = dict()
        for edge_t in self.metadata[1]:
            edge_label_index = data[edge_t].edge_label_index
            num_edges = len(edge_label_index)

            head = out_dict[edge_t[0]][edge_label_index[0]]  # embedding src nodes
            head_re_a = head[:, 0]
            head_im_a = head[:, 1]

            tail = out_dict[edge_t[2]][edge_label_index[1]]  # embedding dst nodes
            tail_re_a = tail[:, 0]
            tail_im_a = tail[:, 1]

            reli = self.rel_to_index[edge_t]
            rel = self.rel_emb[reli]
            rel_re = rel[0]
            rel_im = rel[1]

            # ComplEx score
            h = torch.Tensor(
                [
                    triple_dot(head_re, rel_re, tail_re)
                    + triple_dot(head_im, rel_re, tail_im)
                    + triple_dot(head_re, rel_im, tail_im)
                    - triple_dot(head_im, rel_im, tail_re)
                    for head_re, head_im, tail_re, tail_im in zip(
                        head_re_a, head_im_a, tail_re_a, tail_im_a
                    )
                ]
            )

            h_dict[edge_t] = h

        return h_dict, current_dict_1, current_dict_2

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)
