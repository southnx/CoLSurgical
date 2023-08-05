import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.geometric as pyg
from models.args import get_args

# args = get_args()
# device = torch.device('cuda', args.local_rank)

class MLPUpdater(nn.Module):
    """
    Node embedding update block using simple MLP.
    embedding_new <- MLP([embedding_old, node_feature_new])
    """

    def __init__(self, dim_in, dim_out, layer_id, num_layers):
        super(MLPUpdater, self).__init__()
        self.layer_id = layer_id
        # FIXME:
        assert num_layers > 1, 'There is a problem with layer=1 now, pending fix.'
        self.mlp = MLP(dim_in=dim_in + dim_out, dim_out=dim_out,
                       num_layers=num_layers)

    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        concat = torch.cat((H_prev, X), axis=1)
        H_new = self.mlp(concat)
        batch.node_states[self.layer_id] = H_new
        return H_new


class GRUUpdater(nn.Module):
    """
    Node embedding update block using standard GRU and variations of it.
    """

    def __init__(self, dim_in, dim_out, layer_id=None):
        # dim_in (dim of X): dimension of input node_feature1.
        # dim_out (dim of H): dimension of previous and current hidden states.
        # forward(X, H) --> H.
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layer_id = layer_id
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid()).cuda()
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid()).cuda()
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh()).cuda()

        # self.GRU_Z =nn.Linear(dim_in + dim_out, dim_out, bias=True)
        # # reset gate.
        # self.GRU_R = nn.Linear(dim_in + dim_out, dim_out, bias=True)
        # # new embedding gate.
        # self.GRU_H_Tilde = nn.Linear(dim_in + dim_out, dim_out, bias=True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, H_prev, node_feature):
        r"""
        H_prev: previous node features
        node_feature: current feature
        """
        # assert self.dim_in == node_feature.shape[0]
        # assert self.dim_out == H_prev.shape[0]
        # print("Embedding updating...")
        Z = self.GRU_Z(torch.cat([node_feature, H_prev], dim=1))
        # Z = self.sigmoid(Z)
        R = self.GRU_R(torch.cat([node_feature, H_prev], dim=1))
        # R = self.sigmoid(R)
        H_tilde = self.GRU_H_Tilde(torch.cat([node_feature, R * H_prev], dim=1))
        # H_tilde = self.sigmoid(H_tilde)
        H_gru = Z * H_prev + (1 - Z) * H_tilde
        return H_gru


    # def forward(self, batch):
    #     H_prev = batch.node_states[self.layer_id]
    #     X = batch.node_feature
    #     Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
    #     R = self.GRU_R(torch.cat([X, H_prev], dim=1))
    #     H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
    #     H_gru = Z * H_prev + (1 - Z) * H_tilde

    #     if cfg.gnn.embed_update_method == 'masked_gru':
    #         # Update for active nodes only, use output from GRU.
    #         keep_mask = (batch.node_degree_new == 0)
    #         H_out = H_gru
    #         # Reset inactive nodes' embedding.
    #         H_out[keep_mask, :] = H_prev[keep_mask, :]
    #     elif cfg.gnn.embed_update_method == 'moving_average_gru':
    #         # Only update for active nodes, using moving average with output from GRU.
    #         H_out = H_prev * batch.keep_ratio + H_gru * (1 - batch.keep_ratio)
    #     elif cfg.gnn.embed_update_method == 'gru':
    #         # Update all nodes' embedding using output from GRU.
    #         H_out = H_gru
    #     return H_out
