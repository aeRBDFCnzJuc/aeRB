
import torch
import torch.nn as nn
from deepctr_torch.inputs import DenseFeat,SparseFeat, get_feature_names,build_input_features
from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN


class Wide(torch.nn.Module):
    def __init__(self, config):
        dnn_feature_columns = config.dnn_feature_columns
        num_labels = config.num_labels
        feature_index = config.feature_index
        device = config.device

        dnn_hidden_units = config.wide_dnn_hidden_units if "wide_dnn_hidden_units" in config else  (256, 128)
        dnn_dropout = config.wide_dnn_dropput if "wide_dnn_dropput" in config else 0

        '''model configs
        '''
        super().__init__()
        self.feature_index = feature_index
        self.device = device
        self.dnn_feature_columns = dnn_feature_columns
        self.embedding_dict = create_embedding_matrix(
            dnn_feature_columns, sparse=False)

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       dropout_rate=dnn_dropout)
        self.out = nn.Linear(dnn_hidden_units[-1], num_labels,bias=config.wide_use_bias)

    def forward(self, x):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_logit = self.dnn(dnn_input)
        y_pred = self.out(dnn_logit)
        return y_pred

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(
                feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list
    
    
class WideFlatten(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        dnn_feature_columns = config.dnn_feature_columns
        feature_index = config.feature_index
        device = config.device

        '''model configs
        '''
        self.feature_index = feature_index
        self.device = device
        self.dnn_feature_columns = dnn_feature_columns
        self.embedding_dict = create_embedding_matrix(
            dnn_feature_columns, sparse=False)
        self.output_dim = self.compute_input_dim(
            config.dnn_feature_columns)
        
    def forward(self, x):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        return dnn_input
    
    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
        filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
        feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(
                feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list
    
from area_attention import AreaAttention,MultiHeadAreaAttention

class Deep(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dnn_hidden_units = config.deep_dnn_hidden_units if "deep_dnn_hidden_units" in config else  (256, 128)
        dnn_dropout = config.deep_dnn_dropput if "deep_dnn_dropput" in config else 0
        self.bilstm = nn.LSTM(input_size=config.deep_feature_dim,
                              num_layers=config.lstm_layer_num,
                              hidden_size=config.max_timesteps,
                              bidirectional=config.use_bidirectional,
                              proj_size=config.lstm_output_dim,
                              batch_first=True)

        input_dim = config.lstm_output_dim * \
            2 if config.use_bidirectional else config.lstm_output_dim
        if config.use_attention:
            self.self_attn = nn.MultiheadAttention(
                input_dim, num_heads=1, batch_first=True)
        if config.area_attention:
            area_attn_core = AreaAttention(
                key_query_size=input_dim,
                area_key_mode=config.area_key_mode,
                area_value_mode=config.area_value_mode,
                max_area_width=config.max_area_width,
                memory_width=config.max_len,
                dropout_rate=0.2,
            )
            self.area_attn = MultiHeadAreaAttention(
                area_attention=area_attn_core,
                num_heads=config.num_area_attention_heads,
                key_query_size=input_dim,
                key_query_size_hidden=input_dim,
                value_size=input_dim,
                value_size_hidden=input_dim
            )
        self.dnn = DNN(input_dim, dnn_hidden_units, dropout_rate=dnn_dropout)
        self.out = nn.Linear(dnn_hidden_units[-1], config.num_labels,bias=config.deep_use_bias)

    def forward(self, x):
        x_out, (h_n, c_n) = self.bilstm(x)
#         print("x_out.shape is ",x_out.shape)
        if self.config.use_attention:
            if self.config.area_attention:
                x_out = self.area_attn(x_out, x_out, x_out)
            else:
                x_out = self.self_attn(x_out, x_out, x_out, need_weights=False)[0]
#             print("attention x_out is ",x_out)
            x_out = torch.mean(x_out, axis=1)
        x_out = self.dnn(x_out)
        out = self.out(x_out)
        return out
    
    
class DeepFlatten(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dnn_hidden_units = config.deep_dnn_hidden_units if "deep_dnn_hidden_units" in config else  (256, 128)
        dnn_dropout = config.deep_dnn_dropput if "deep_dnn_dropput" in config else 0
        self.bilstm = nn.LSTM(input_size=config.deep_feature_dim,
                              num_layers=config.lstm_layer_num,
                              hidden_size=config.max_timesteps,
                              bidirectional=config.use_bidirectional,
                              proj_size=config.lstm_output_dim,
                              batch_first=True)

        input_dim = config.lstm_output_dim * \
            2 if config.use_bidirectional else config.lstm_output_dim
        if config.use_attention:
            self.self_attn = nn.MultiheadAttention(
                input_dim, num_heads=1, batch_first=True)
        if config.area_attention:
            area_attn_core = AreaAttention(
                key_query_size=input_dim,
                area_key_mode=config.area_key_mode,
                area_value_mode=config.area_value_mode,
                max_area_width=config.max_area_width,
                memory_width=config.max_len,
                dropout_rate=0.2,
            )
            self.area_attn = MultiHeadAreaAttention(
                area_attention=area_attn_core,
                num_heads=config.num_area_attention_heads,
                key_query_size=input_dim,
                key_query_size_hidden=input_dim,
                value_size=input_dim,
                value_size_hidden=input_dim
            )
        self.output_dim = input_dim
        
    def forward(self, x):
        x_out, (h_n, c_n) = self.bilstm(x)
#         print("x_out.shape is ",x_out.shape)
        if self.config.use_attention:
            if self.config.area_attention:
                x_out = self.area_attn(x_out, x_out, x_out)
            else:
                x_out = self.self_attn(x_out, x_out, x_out, need_weights=False)[0]
#             print("attention x_out is ",x_out)
            x_out = torch.mean(x_out, axis=1)
        return x_out
    
    
class WideDeepEF(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # wide
        self.wide = WideFlatten(config)
        wide_output_dim = self.wide.output_dim
        # deep nets
        self.deep = DeepFlatten(config)
        # wide&deep nets
        deep_output_dim = self.deep.output_dim
        self.dnn = DNN(wide_output_dim+deep_output_dim, config.dnn_hidden_units,dropout_rate=config.dropout)
        self.out = nn.Linear(config.dnn_hidden_units[-1], config.num_labels)
        
    def forward(self, x_wide, x_deep):
            # wide
        wide_output = self.wide(x_wide)
        # deep
        deep_output = self.deep(x_deep)
        # print("deep_output shape is",deep_output.shape)
        # wide&deep
        logit = torch.cat([wide_output, deep_output], dim=1)
        logit = self.dnn(logit)
        logit = self.out(logit)
        return logit,logit,logit
    
    
from .wide_models import PNNFlatten
class PNNDeepEF(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # wide
        self.wide = PNNFlatten(config)
        # deep nets
        self.deep = DeepFlatten(config)
        # wide&deep nets

        self.dnn = DNN(self.wide.output_dim+self.deep.output_dim, config.dnn_hidden_units,dropout_rate=config.dropout)
        self.out = nn.Linear(config.dnn_hidden_units[-1], config.num_labels)
        
    def forward(self, x_wide, x_deep):
            # wide
        wide_output = self.wide(x_wide)
        # print("wide_output shape is",wide_output.shape,self.wide.output_dim)
        # deep
        deep_output = self.deep(x_deep)
        # print("deep_output shape is",deep_output.shape,self.deep.output_dim)
        # wide&deep
        logit = torch.cat([wide_output, deep_output], dim=-1)
        logit = self.dnn(logit)
        logit = self.out(logit)
        return logit,logit,logit