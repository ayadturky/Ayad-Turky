#Some code references chemicalx(https://github.com/AstraZeneca/chemicalx).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, SAGPooling
from torch_geometric.nn import global_max_pool , global_mean_pool,global_add_pool
import numpy as np
from torch_geometric.nn import MessagePassing


################################################Ayad########################################

class EnhancedDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.3, output_size=2):
        super(EnhancedDNN, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MPNNSyn(nn.Module):
    def __init__(
        self,
        molecule_channels: int = 78,
        hidden_channels: int = 128,
        layer_count: int = 2,
        out_channels: int = 2,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.graph_convolutions = torch.nn.ModuleList()

        # Initialize MPNN layers
        self.graph_convolutions.append(CustomMPNN(molecule_channels, hidden_channels))
        for _ in range(1, layer_count):
            self.graph_convolutions.append(CustomMPNN(hidden_channels, hidden_channels))

        self.border_rnn = torch.nn.LSTM(hidden_channels, hidden_channels, 1)

        self.reduction = nn.Sequential(
            nn.Linear(954, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.reduction2 = nn.Sequential(
            nn.Linear(954, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 78),
            nn.ReLU()
        )
        self.pool1 = Attention(hidden_channels, 4)
        self.pool2 = Attention(hidden_channels, 4)

        # Enhanced DNN as the prediction module
        self.final = EnhancedDNN(
            input_size=4 * hidden_channels + 256,
            hidden_sizes=[512, 256, 128],  # Flexible hidden layer sizes
            dropout_rate=0.3,
            output_size=out_channels
        )

    def _forward_molecules(
        self,
        conv,
        x,
        edge_index,
        batch,
        states,
    ) -> torch.FloatTensor:
        gcn_hidden = conv(x, edge_index)
        gcn_hidden_detach = gcn_hidden.detach()
        rnn_out, (hidden_state, cell_state) = self.border_rnn(gcn_hidden_detach[None, :, :], states)
        rnn_out = rnn_out.squeeze()
        return gcn_hidden, rnn_out, (hidden_state, cell_state)

    def forward(self, molecules_left, molecules_right) -> torch.FloatTensor:
        x1, edge_index1, batch1, cell, mask1 = molecules_left.x, molecules_left.edge_index, molecules_left.batch, molecules_left.cell, molecules_left.mask
        x2, edge_index2, batch2, mask2 = molecules_right.x, molecules_right.edge_index, molecules_right.batch, molecules_right.mask

        cell = F.normalize(cell, 2, 1)
        cell_expand = self.reduction2(cell)
        cell = self.reduction(cell)

        cell_expand = cell_expand.unsqueeze(1)
        cell_expand = cell_expand.expand(cell.shape[0], 100, -1)
        cell_expand = cell_expand.reshape(-1, 78)

        batch_size = torch.max(molecules_left.batch) + 1
        mask1 = mask1.reshape((batch_size, 100))
        mask2 = mask2.reshape((batch_size, 100))

        left_states, right_states = None, None
        gcn_hidden_left = molecules_left.x + cell_expand
        gcn_hidden_right = molecules_right.x + cell_expand

        for conv in self.graph_convolutions:
            gcn_hidden_left, rnn_out_left, left_states = self._forward_molecules(
                conv, gcn_hidden_left, molecules_left.edge_index, molecules_left.batch, left_states
            )
            gcn_hidden_right, rnn_out_right, right_states = self._forward_molecules(
                conv, gcn_hidden_right, molecules_right.edge_index, molecules_right.batch, right_states
            )

        rnn_out_left, rnn_out_right = rnn_out_left.reshape(batch_size, 100, -1), rnn_out_right.reshape(batch_size, 100, -1)
        rnn_pooled_left, rnn_pooled_right = self.pool1(rnn_out_left, rnn_out_right, (mask1, mask2))

        gcn_hidden_left, gcn_hidden_right = gcn_hidden_left.reshape(batch_size, 100, -1), gcn_hidden_right.reshape(batch_size, 100, -1)
        gcn_hidden_left, gcn_hidden_right = self.pool2(gcn_hidden_left, gcn_hidden_right, (mask1, mask2))
        shared_graph_level = torch.cat([gcn_hidden_left, gcn_hidden_right], dim=1)

        out = torch.cat([shared_graph_level, rnn_pooled_left, rnn_pooled_right, cell], dim=1)
        out = self.final(out)
        return out
    def extract_embeddings(self, molecules_left, molecules_right):
        """
        Extract embeddings from MPNNSyn before the final prediction layer.
        """
        x1, edge_index1, batch1, cell, mask1 = molecules_left.x, molecules_left.edge_index, molecules_left.batch, molecules_left.cell, molecules_left.mask
        x2, edge_index2, batch2, mask2 = molecules_right.x, molecules_right.edge_index, molecules_right.batch, molecules_right.mask
    
        cell = F.normalize(cell, 2, 1)
        cell_expand = self.reduction2(cell)
        cell = self.reduction(cell)
    
        cell_expand = cell_expand.unsqueeze(1)
        cell_expand = cell_expand.expand(cell.shape[0], 100, -1)
        cell_expand = cell_expand.reshape(-1, 78)
    
        batch_size = torch.max(molecules_left.batch) + 1
        mask1 = mask1.reshape((batch_size, 100))
        mask2 = mask2.reshape((batch_size, 100))
    
        left_states, right_states = None, None
        gcn_hidden_left = molecules_left.x + cell_expand
        gcn_hidden_right = molecules_right.x + cell_expand
    
        for conv in self.graph_convolutions:
            gcn_hidden_left, _, left_states = self._forward_molecules(
                conv, gcn_hidden_left, molecules_left.edge_index, molecules_left.batch, left_states
            )
            gcn_hidden_right, _, right_states = self._forward_molecules(
                conv, gcn_hidden_right, molecules_right.edge_index, molecules_right.batch, right_states
            )
    
        gcn_hidden_left, gcn_hidden_right = gcn_hidden_left.reshape(batch_size, 100, -1), gcn_hidden_right.reshape(batch_size, 100, -1)
        pooled_left, pooled_right = self.pool1(gcn_hidden_left, gcn_hidden_right, (mask1, mask2))
        shared_embedding = torch.cat([pooled_left, pooled_right], dim=1)
    
        return shared_embedding



class CustomMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # Use "add" aggregation

        # Linear layer for projecting x_j to out_channels
        self.lin = nn.Linear(in_channels, out_channels)

        # MLP for edge attributes
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=None)

    def message(self, x_j, x_i):
        # Compute edge attributes
        edge_attr = self.edge_mlp(torch.cat([x_i, x_j], dim=-1))
        
        # Project x_j to match edge_attr dimensions
        x_j_projected = self.lin(x_j)
        
        return x_j_projected + edge_attr



class Attention(nn.Module):
    def __init__(self,  dim, num_heads = 4):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.linear_q = nn.Linear(dim, self.dim_per_head*num_heads)
        self.linear_k = nn.Linear(dim, self.dim_per_head*num_heads)
        self.linear_v = nn.Linear(dim, self.dim_per_head*num_heads)
        #self.norm = nn.BatchNorm1d(dim)
        self.norm = nn.LayerNorm(dim)
        self.linear_final = nn.Linear(dim,dim)

        self.linear_q_inner = nn.Linear(dim, dim)
        self.linear_k_inner = nn.Linear(dim, dim)
        self.linear_v_inner = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(p=0.2)




    def attention(self, q1,k1,v1,q2,k2,v2,attn_mask=None,flag=False):
        #print('k1',k1[0])
        #print(True in torch.isnan(k1[0]))
        #print('q1',q2[0])
        #print(True in torch.isnan(q1[0]))

        a1 = torch.tanh(torch.bmm(k1, q2.transpose(1, 2)))
        a2 = torch.tanh(torch.bmm(k2, q1.transpose(1, 2)))
        #print(a1[0])
        if attn_mask is not None:
            #a1=a1.masked_fill(attn_mask, -np.inf)
            #a2=a2.masked_fill(attn_mask.transpose(1, -1), -np.inf)
            mask1 = attn_mask[0]
            mask2 = attn_mask[1]

            a1 = torch.softmax(torch.sum(a1, dim=2).masked_fill(mask1, -np.inf), dim=-1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2).masked_fill(mask2, -np.inf), dim=-1).unsqueeze(dim=1)
        else:
            a1 = torch.softmax(torch.sum(a1, dim=2), dim=1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2), dim=1).unsqueeze(dim=1)
            #print('after softmax',a1[0])

        a1 = self.dropout(a1)
        a2 = self.dropout(a2)
        #print(a1.shape, v1.shape, mask1.shape)

        vector1 = torch.bmm(a1, v1).squeeze()
        vector2 = torch.bmm(a2, v2).squeeze()

        return vector1,vector2

    def forward(self, fingerprint_vectors1,  fingerprint_vectors2, attn_mask=None, flag=False):
        #batch_size = fingerprint_vectors1.shape[0]
        #fingerprint_vectors1 = self.self_attention(fingerprint_vectors1)
        #fingerprint_vectors2 = self.self_attention(fingerprint_vectors2)


        q1, q2 = torch.relu(self.linear_q(fingerprint_vectors1)), torch.relu(self.linear_q(fingerprint_vectors2))
        k1, k2 = torch.relu(self.linear_k(fingerprint_vectors1)), torch.relu(self.linear_k(fingerprint_vectors2))
        v1, v2 = torch.relu(self.linear_v(fingerprint_vectors1)), torch.relu(self.linear_v(fingerprint_vectors2))
        '''
        q1, q2 = fingerprint_vectors1,fingerprint_vectors2
        k1, k2 = fingerprint_vectors1, fingerprint_vectors2
        v1, v2 = fingerprint_vectors1, fingerprint_vectors2
        '''
        vector1, vector2 = self.attention(q1,k1,v1,q2,k2,v2, attn_mask, flag)


        vector1 = self.norm(torch.mean(fingerprint_vectors1, dim=1) + vector1)
        vector2 = self.norm(torch.mean(fingerprint_vectors2, dim=1) + vector2)


        return vector1, vector2


