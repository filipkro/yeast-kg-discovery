from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
import torch as th
import torch.nn as nn

import torch.nn.functional as F


DROP_OUT = 0.0


class TargetChemicalResidualHead(nn.Module):
    """A zero-initialised, gene-specific correction for one chemical."""
    def __init__(self, input_dim, hidden_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_channels:
            self.layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            prev_dim = hidden_dim
        self.output = nn.Linear(prev_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # A zero residual preserves the original shared-decoder behaviour at
        # initialisation; each specialist is learned only when its chemical is
        # observed during training.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        return self.output(x)


class OGGNNBase(th.nn.Module):
    def __init__(self):
        print("OGGNNBase")
        super().__init__()
        self.layers = th.nn.ModuleList()

    def reset_parameters(self):
        for layer in self.layers:
            for rel in layer.convs.values():
                for k, module in rel._modules.items():
                    if 'lin' in k:
                        nn.init.xavier_uniform_(module.weight)
            # pass
        
    def init_gnn(self, channels, embedding_dims, edge_types, es, skip_last=True):
        prev_c = 0
        for i, c in enumerate(channels):
            conv_dict = {}
            for e in edge_types:

                if i == len(channels) - 1 and e[2] not in ['genes', 'mat_ent']:
                    continue


                source_channels = int(i == 0) * embedding_dims[e[0]] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[0]])))
                target_channels = int(i == 0) * embedding_dims[e[2]] + int(
                    i > 0
                ) * max((1, int(prev_c * es[e[2]])))
                out_channels = max((1, int(c * es[e[2]])))
                root_weight = True

                conv_dict[e] = SAGEConv(
                    (source_channels, target_channels),
                    out_channels,
                    normalize=True,
                    root_weight=root_weight,
                    project=True,  # aggr='lstm')
                    aggr="max",
                )
            conv = HeteroConv(conv_dict, aggr="mean")
            prev_c = c
            self.layers.append(conv)

    def forward(self, x_dict, edge_index_dict, return_embs=False):
        embs = [x_dict]
        for conv in self.layers:
            x_dict = conv(x_dict, edge_index_dict)
            if return_embs:
                embs.append(x_dict)
        if return_embs:
            return embs
        return x_dict

class OGGNNCustom(OGGNNBase):
    def __init__(self, channels, edge_types, embedding_dims, skip_last=True):
        print("OGGNNCustom")
        super().__init__()
        ed = {k: 0 for k in embedding_dims.keys()}
        for e, v in edge_types.items():
            ed[e[0]] += v
            ed[e[2]] += v
        self.es = {}
        print(ed)
        for k, v in ed.items():
            if v / 500000 > 1:
                self.es[k] = 2
            elif v / 100000 > 1:
                self.es[k] = 1
            elif v / 10000 > 1:
                self.es[k] = 0.5
            else:
                self.es[k] = 0.25

        self.init_gnn(channels, embedding_dims, edge_types, self.es, skip_last=skip_last)


class OGGNN(OGGNNBase):
    def __init__(self, channels, edge_types, embedding_dims, skip_last=False):
        print("OGGNN")
        super().__init__()
        self.layers = th.nn.ModuleList()
        self.es = {k: 1 for k in embedding_dims.keys()}
        self.init_gnn(channels, embedding_dims, edge_types, self.es, skip_last=skip_last)


class Model(th.nn.Module):
    def __init__(
        self,
        gnn_channels: list,
        nn_channels: list,
        embedding_dims,
        number_classes,
        message_edge_types=[],
        predicted_edge_types=None,
        custom=True,
        aggr="attn",
        target_chemical_indices=None,
        target_head_channels=None,
        target_node_type='mat_ent',
        device='cpu'
    ):
        super().__init__()
        if not predicted_edge_types:
            predicted_edge_types = message_edge_types
        # custom = False
        # print('none custom sizes for gnn')

        CUSTOM_OGGNN = custom
        if len(gnn_channels) > 0:
            self.node_embeddings = th.nn.ModuleDict(
                {k: th.nn.Embedding(num_embeddings=number_classes[k],
                                    embedding_dim=v) for k,v in
                 embedding_dims.items()})
            self.dropout = th.nn.Dropout(DROP_OUT)

            self.gnn = OGGNN(channels=gnn_channels, edge_types=message_edge_types,
                                 embedding_dims=embedding_dims, skip_last=False)


            prev_width = gnn_channels[-1]
            prev_width = prev_width * 2
            first_width = prev_width
            if len(nn_channels) > 0:
                self.lin_layers = th.nn.ModuleDict()
                for k in predicted_edge_types:
                    prev_width = first_width
                    layers = []
                    for c in nn_channels:
                        layers.append(th.nn.Linear(prev_width, c, bias=True))
                        prev_width = c
                    self.lin_layers["_".join(k)] = th.nn.ModuleList(layers)
            else:
                self.lin_layers = None
        last_layer_in = nn_channels[-1] if self.lin_layers else first_width
        self.last_layer = th.nn.ModuleDict({"_".join(k): th.nn.Linear(last_layer_in, 1, bias=True) for k in predicted_edge_types})
        if target_chemical_indices is None:
            target_chemical_indices = []
        elif isinstance(target_chemical_indices, dict):
            target_chemical_indices = target_chemical_indices.values()
        self.target_chemical_indices = tuple(
            sorted({int(index) for index in target_chemical_indices})
        )
        self.target_node_type = target_node_type
        if target_head_channels is None:
            target_head_channels = nn_channels
        self.target_heads = nn.ModuleList([
            TargetChemicalResidualHead(gnn_channels[-1], target_head_channels)
            for _ in self.target_chemical_indices
        ])
        self._neighbors_to_sample = None
        self.device = device

    def reset_parameters(self):
        for embedding in self.node_embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)
        for decoder in self.lin_layers.values():
            for layer in decoder:
                nn.init.xavier_uniform_(layer.weight)
        for layer in self.last_layer.values():
            nn.init.xavier_uniform_(layer.weight)
        for target_head in self.target_heads:
            target_head.reset_parameters()
        self.gnn.reset_parameters()
        # nn.init.xavier_uniform_(layer.weight)

    @property
    def neighbors_to_sample(self):
        if self._neighbors_to_sample == None:
            raise AttributeError("neighbors_to_sample is not set")
        else:
            return self._neighbors_to_sample

    def set_neighbors_to_sample(self, neighbors, val_neighbors=None):
        if val_neighbors == None:
            val_neighbors = neighbors
        self._neighbors_to_sample = {
            "neighbors": neighbors,
            "val_neighbors": val_neighbors,
        }

    def forward(self, data: HeteroData):
        raise NotImplementedError()
    
    def _get_embeddings(self, data: HeteroData, return_embs=False):
        x_dict = {
            k: self.node_embeddings[k](data[k].node_id) for k in self.node_embeddings
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict, return_embs=return_embs)

        return x_dict

    def _forward(self, data: HeteroData, edges_to_predict=None, return_embs=False) -> th.Tensor:
        # links_to_pred = data[LINKS].edge_label_index
        if not edges_to_predict:
            edges_to_predict = data.edge_label_index_dict
        
        x_dict = self._get_embeddings(data, return_embs=return_embs)
        embs = x_dict[-1] if return_embs else x_dict

        all_z = th.tensor([], device=self.device)
        for k, v in edges_to_predict.items():
            h = embs[k[0]][v[0]]
            t = embs[k[2]][v[1]]
        
            pair_embeddings = th.cat([h, t], dim=-1)
            z = pair_embeddings

            if self.lin_layers:
                for l in self.lin_layers["_".join(k)]:
                    z = l(z).relu()
                    z = self.dropout(z)
                z = self.last_layer["_".join(k)](z)
            else:
                z = self.last_layer["_".join(k)](z)

            # All edges retain the global decoder score.  A target-specific
            # residual is added only for configured chemical node indices.
            # Computing each small head over the batch keeps the operation
            # tensorised and compatible with torch.compile.
            if self.target_heads and k[2] == self.target_node_type:
                target_residual = th.zeros_like(z)
                for chemical_index, target_head in zip(
                        self.target_chemical_indices, self.target_heads):
                    target_mask = (v[1] == chemical_index).unsqueeze(-1)
                    target_residual = target_residual + (target_head(h) * target_mask)
                z = z + target_residual
            all_z = th.cat((all_z, z), dim=0)
        

        if return_embs:
            return all_z, x_dict
        else:
            return all_z

class Classifier(Model):
    def __init__(
        self,
        gnn_channels: list,
        nn_channels: list,
        embedding_dims,
        number_classes,
        message_edge_types,
        predicted_edge_types=None,
        custom=True,
        target_chemical_indices=None,
        target_head_channels=None,
        target_node_type='mat_ent',
        device='cpu'
    ):
        super().__init__(gnn_channels=gnn_channels, nn_channels=nn_channels,
                         embedding_dims=embedding_dims, message_edge_types=message_edge_types,
                         predicted_edge_types=predicted_edge_types,
                         number_classes=number_classes, custom=custom,
                         target_chemical_indices=target_chemical_indices,
                         target_head_channels=target_head_channels,
                         target_node_type=target_node_type,
                         device=device)
        # self.activation = th.nn.Sigmoid()
        self.activation = th.nn.Identity()

        # if len(nn_channels) > 0:
        #     self.last_lin = th.nn.Linear(nn_channels[-1], 1)
        # else:
        #     self.last_lin = th.nn.Linear(1, 1)

    # def reset_parameters(self)

    def forward(self, data: HeteroData, edges_to_predict=None, return_embs=False):
        if return_embs:
            z, x_dicts = self._forward(data, edges_to_predict=edges_to_predict, return_embs=return_embs)
            # return self.activation(self.last_lin(z)), x_dicts
            return self.activation(z), x_dicts
        else:
            z = self._forward(data, edges_to_predict=edges_to_predict, return_embs=False)
            # return self.activation(self.last_lin(z))
            return self.activation(z)
        
    def get_embeddings(self, data: HeteroData, return_embs=True):
        return self._get_embeddings(data, return_embs=return_embs)


class MultiDomainBase(nn.Module):
    def __init__(
        self,
        gnn_channels: list,
        embedding_dims: dict,
        number_classes: dict,
        message_edge_types: list,
        hidden_dim=128,  # complex dim (real+imag = 2*hidden_dim)
        use_layernorm=True,
        device='cpu'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.device = device
        self.node_embeddings = th.nn.ModuleDict(
                {k: th.nn.Embedding(num_embeddings=number_classes[k],
                                    embedding_dim=v) for k,v in
                 embedding_dims.items()})
        self.gnn = OGGNN(channels=gnn_channels, edge_types=message_edge_types,
                         embedding_dims=embedding_dims, skip_last=False)
        # Domain-specific projections → shared complex space
        # might be worth investigating if domain-rel specific projections improves
        self.proj = nn.ModuleDict({
            domain: nn.Linear(gnn_channels[-1], 2 * hidden_dim, device=device)
            for domain in embedding_dims
        })

        self.norm = nn.ModuleDict({
            domain: nn.LayerNorm(2 * hidden_dim, device=device)
            for domain in embedding_dims
        }) if use_layernorm else None

    def reset_parameters(self):
        for layer in self.proj.values():
            nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def split_complex(self, x):
        """Split real and imaginary parts"""
        return x[..., :self.hidden_dim], x[..., self.hidden_dim:]

    def project(self, h, domain):
        h = self.proj[domain](h)
        if self.norm is not None:
            h = self.norm[domain](h)
        return h




class MultiDomainComplEx(MultiDomainBase):
    def __init__(
        self,
        gnn_channels: list,
        embedding_dims: dict,
        number_classes: dict,
        message_edge_types=[],
        predicted_edge_types=None,
        hidden_dim=128,  # complex dim (real+imag = 2*hidden_dim)
        use_layernorm=True,
        device='cpu'
    ):
        super().__init__(
            gnn_channels=gnn_channels,
            embedding_dims=embedding_dims,
            number_classes=number_classes,
            message_edge_types=message_edge_types,
            hidden_dim=hidden_dim,  # complex dim (real+imag = 2*hidden_dim)
            use_layernorm=use_layernorm,
            device=device
        )

        # Relation embeddings (complex)
        self.rel_ids = {k[1]: i for i, k in enumerate(predicted_edge_types)}
        self.rel = nn.Embedding(len(predicted_edge_types), 2 * hidden_dim,
                                device=device)

        self.reset_parameters()


    def score(self, h_u, h_v, r):
        """ComplEx scoring"""
        u_re, u_im = self.split_complex(h_u)
        v_re, v_im = self.split_complex(h_v)
        r_re, r_im = self.split_complex(r)

        return (
            u_re * r_re * v_re +
            u_im * r_re * v_im +
            u_re * r_im * v_im -
            u_im * r_im * v_re
        ).sum(dim=-1)

    def forward(self, data: HeteroData, edges_to_predict=None, return_embs=False):
        """
        h_u: tensor [batch, d_u]
        h_v: tensor [batch, d_v]
        rel_ids: tensor [batch]
        domain_u/domain_v: string (or list if batching mixed types)
        """

        if not edges_to_predict:
            edges_to_predict = data.edge_label_index_dict
        x_dict = {
            k: self.node_embeddings[k](data[k].node_id) for k in self.node_embeddings
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict, return_embs=return_embs)
        embs = x_dict[-1] if return_embs else x_dict

        heads = th.tensor([], device=self.device)
        tails = th.tensor([], device=self.device)
        rels = th.tensor([], device=self.device)
        for k, v in edges_to_predict.items():
            h = self.project(embs[k[0]][v[0]], k[0])
            t = self.project(embs[k[2]][v[1]], k[2])
            r = self.rel(th.ones((len(h),), dtype=th.long, device=self.device)
                         * self.rel_ids[k[1]])

            heads = th.cat((heads, h), dim=0)
            tails = th.cat((tails, t), dim=0)
            rels = th.cat((rels, r), dim=0)

        if return_embs:
            return self.score(heads, tails, rels), x_dict
        else:
            return self.score(heads, tails, rels)


class MultiDomainRotatE(MultiDomainBase):
    def __init__(
        self,
        gnn_channels: list,
        embedding_dims: dict,
        number_classes: dict,
        message_edge_types=[],
        predicted_edge_types=None,
        hidden_dim=128,
        gamma=12.0,
        use_layernorm=True,
        device='cpu'
    ):
        super().__init__(
            gnn_channels=gnn_channels,
            embedding_dims=embedding_dims,
            number_classes=number_classes,
            message_edge_types=message_edge_types,
            hidden_dim=hidden_dim,  # complex dim (real+imag = 2*hidden_dim)
            use_layernorm=use_layernorm,
            device=device
        )

        self.rel_ids = {k[1]: i for i, k in enumerate(predicted_edge_types)}
        self.rel = nn.Embedding(
            len(predicted_edge_types),
            hidden_dim,
            device=device
        )

        
        self.gamma = nn.Parameter(th.tensor(gamma, device=device))

        self.reset_parameters()

    def score(self, h_u, h_v, phase):
        """
        RotatE scoring:
        rotate h_u by relation, compare to h_v
        """
        u_re, u_im = self.split_complex(h_u)
        v_re, v_im = self.split_complex(h_v)

        r_re = th.cos(phase)
        r_im = th.sin(phase)

        # rotate head
        rot_u_re = u_re * r_re - u_im * r_im
        rot_u_im = u_re * r_im + u_im * r_re

        # distance
        re_diff = rot_u_re - v_re
        im_diff = rot_u_im - v_im

        dist = th.sqrt(re_diff**2 + im_diff**2 + 1e-9)
        gamma = F.softplus(self.gamma)
        return gamma - dist.sum(dim=-1)

    def forward(self, data: HeteroData, edges_to_predict=None, return_embs=False):

        if not edges_to_predict:
            edges_to_predict = data.edge_label_index_dict

        x_dict = {
            k: self.node_embeddings[k](data[k].node_id)
            for k in self.node_embeddings
        }

        x_dict = self.gnn(x_dict, data.edge_index_dict, return_embs=return_embs)
        embs = x_dict[-1] if return_embs else x_dict

        heads = th.empty((0, 2 * self.hidden_dim), device=self.device)
        tails = th.empty((0, 2 * self.hidden_dim), device=self.device)
        phases = th.empty((0, self.hidden_dim), device=self.device)

        for k, v in edges_to_predict.items():
            h = self.project(embs[k[0]][v[0]], k[0])
            t = self.project(embs[k[2]][v[1]], k[2])

            rel_id = self.rel_ids[k[1]]
            phase = self.rel(
                th.full((len(h),), rel_id, dtype=th.long, device=self.device)
            )

            heads = th.cat((heads, h), dim=0)
            tails = th.cat((tails, t), dim=0)
            phases = th.cat((phases, phase), dim=0)

        scores = self.score(heads, tails, phases)

        if return_embs:
            return scores, x_dict
        else:
            return scores
