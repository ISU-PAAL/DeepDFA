#%%
from sastvd.linevd.graphmogrifier import get_nodes_df
sample_mode = True
feat = "_ABS_DATAFLOW_datatype_all"
label_style = "graph"
partition = "all"
nodes_df = get_nodes_df(sample_mode, feat)
nodes_df

#%%
import sastvd.helpers.codebert as svdc
d2v = None
glove_dict = None
codebert = svdc.CodeBert(cuda=False)

#%%
import sastvd as svd
import torch
import tqdm
node_code = nodes_df.code.astype(str).tolist()
node_code_chunks = svd.chunks(node_code, 128)
node_code_embed = []
for chunk in tqdm.tqdm(node_code_chunks):
    node_code_embed.append(codebert.encode_sents(chunk))
cb_feats = torch.cat(node_code_embed, dim=0)
cb_feats.shape

#%%
node_code = nodes_df.code.astype(str).tolist()
node_code_tok = codebert.tokenize(node_code)
node_code_tok["input_ids"].shape

#%%
import sastvd.helpers.datasets as svdd
df = svdd.bigvul(sample=sample_mode)
code = dict(zip(df.id, df.before))

#%%
from sastvd.linevd.graphmogrifier import get_graphs
additional_features = ["_CODEBERT_INPUT"]
graphs = get_graphs(nodes_df, sample_mode, feat, label_style, partition, additional_features, d2v, glove_dict, codebert, code)
graphs

#%%
from code_gnn.models.flow_gnn.ggnn import FlowGNNGGNNModule
model = FlowGNNGGNNModule(feat,
                input_dim=1002,
                hidden_dim=32,
                n_steps=5,
                num_output_layers=1,
                label_style="graph",
                use_codebert=True)

#%%
model(graphs[0])
