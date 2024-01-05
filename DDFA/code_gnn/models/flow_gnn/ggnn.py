"""
Gated Graph Neural Network module for graph classification tasks
"""
import itertools
from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
import torch
from torch import nn

from code_gnn.models.base_module import BaseModule

from pytorch_lightning.utilities.cli import MODEL_REGISTRY

import logging

logger = logging.getLogger(__name__)

allfeats = [
    "api", "datatype", "literal", "operator",
]

@MODEL_REGISTRY
class FlowGNNGGNNModule(BaseModule):
    def __init__(self,
                feat,
                input_dim,
                hidden_dim,
                n_steps,
                num_output_layers,
                label_style="graph",
                concat_all_absdf=False,
                encoder_mode=False,
                **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if "_ABS_DATAFLOW" in feat:
            feat = "_ABS_DATAFLOW"
        self.feature_keys = {
            "feature": feat,
        }
        
        self.input_dim = input_dim
        self.concat_all_absdf = concat_all_absdf

        # feature extractors
        embedding_dim = hidden_dim  # TODO: try varying embedding dim from hidden_dim
        if self.concat_all_absdf:
            self.all_embeddings = nn.ModuleDict({
                of: nn.Embedding(input_dim, embedding_dim) for of in allfeats
            })
            embedding_dim *= len(allfeats)
            hidden_dim *= len(allfeats)  # TODO: try compressing 4*embeding_dim to hidden_dim
        else:
            self.embedding = nn.Embedding(input_dim, embedding_dim)

        # graph stage
        self.ggnn = GatedGraphConv(in_feats=embedding_dim,
                                out_feats=hidden_dim,
                                n_steps=n_steps,
                                n_etypes=1)

        output_in_size = embedding_dim + hidden_dim

        self.out_dim = output_in_size

        if label_style == "graph":
            pooling_gate_nn = nn.Linear(output_in_size, 1)
            self.pooling = GlobalAttentionPooling(pooling_gate_nn)

        if not encoder_mode:
            output_layers = []
            for i in range(num_output_layers):
                if i == num_output_layers-1:
                    output_size = 1
                else:
                    output_size = output_in_size
                output_layers.append(nn.Linear(output_in_size, output_size))
                if i != num_output_layers-1:
                    output_layers.append(nn.ReLU())
            self.output_layer = nn.Sequential(*output_layers)

    def forward(self, graph, extrafeats):
        # get embedding of feature
        if self.concat_all_absdf:
            cfeats = []
            for otherfeat in allfeats:
                feat = graph.ndata[f"_ABS_DATAFLOW_{otherfeat}"]
                cfeats.append(self.all_embeddings[otherfeat](feat))
            feat_embed = torch.cat(cfeats, dim=1)
        else:
            feat = graph.ndata[self.feature_keys["feature"]]
            feat_embed = self.embedding(feat)

        # graph learning stage
        ggnn_out = self.ggnn(graph, feat_embed)
        
        # concat input
        out = torch.cat([ggnn_out, feat_embed], -1)

        # prediction stage
        if self.hparams.label_style == "graph":
            out = self.pooling(graph, out)

        if self.hparams.encoder_mode:
            logits = out
        else:
            logits = self.output_layer(out).squeeze()

        return logits
