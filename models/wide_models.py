
import torch
import torch.nn as nn
from deepctr_torch.inputs import DenseFeat,SparseFeat
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN


class BaseWideModel(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.feature_index = config.feature_index
        self.dnn_feature_columns = config.dnn_feature_columns
        self.embedding_dict = create_embedding_matrix(config.dnn_feature_columns, sparse=False)
        self.output_dim = self.compute_input_dim(config.dnn_feature_columns)
        
    def forward(self, x):
        raise NotImplementedError
    
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
    

class WideFlatten(BaseWideModel):
    """Return wide component embedding

    Args:
        torch ([type]): [description]
    """
    def __init__(self, config):
        super().__init__(config)
        '''model configs
        '''
        
    def forward(self, x):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        return dnn_input
    

class Wide(WideFlatten):
    """wide embedding + DNN
    Args:
        WideFlatten ([type]): [description]
    """
    def __init__(self, config):
        '''model configs
        '''
        super().__init__(config)
        dnn_hidden_units = config.wide_dnn_hidden_units if "wide_dnn_hidden_units" in config else  (256, 128)
        dnn_dropout = config.wide_dnn_dropput if "wide_dnn_dropput" in config else 0
        self.dnn = DNN(self.compute_input_dim(config.dnn_feature_columns), dnn_hidden_units,
                       dropout_rate=dnn_dropout)
        self.out = nn.Linear(dnn_hidden_units[-1], config.num_labels,bias=config.wide_use_bias)

    def forward(self, x):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_logit = self.dnn(dnn_input)
        y_pred = self.out(dnn_logit)
        return y_pred
    
    
# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.(https://arxiv.org/pdf/1611.00144.pdf)
"""
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, concat_fun, InnerProductLayer, OutterProductLayer


class PNNFlatten(BaseWideModel):
    """Instantiates the Product-based Neural Network architecture.
    config
    :return: A PyTorch model instance.

    """
    def __init__(self, config):
        super().__init__(config)
        kernel_type = "mat"
        if kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")
        self.use_inner = config.use_inner if "use_inner" in config else True
        self.use_outter = config.use_outter if "use_outter" in config else True
        self.kernel_type = kernel_type
        product_out_dim = 0
        num_inputs = self.compute_input_dim(
            self.dnn_feature_columns, include_dense=False, feature_group=True)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)

        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer()

        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(
                num_inputs, config.sparse_emb_dim, kernel_type=kernel_type)
        self.product_out_dim = product_out_dim
        self.output_dim = self.product_out_dim + self.compute_input_dim(self.dnn_feature_columns)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        linear_signal = torch.flatten(
            concat_fun(sparse_embedding_list), start_dim=1)

        if self.use_inner:
            inner_product = torch.flatten(
                self.innerproduct(sparse_embedding_list), start_dim=1)

        if self.use_outter:
            outer_product = self.outterproduct(sparse_embedding_list)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal
#         print(product_layer.shape)
        dnn_input = combined_dnn_input([product_layer], dense_value_list)#concat dense feature
        return dnn_input



class PNN(PNNFlatten):
    """Instantiates the Product-based Neural Network architecture.
    config
    :return: A PyTorch model instance.

    """
    def __init__(self, config):
        super().__init__(config)
        self.pnn = PNNFlatten(config)
        dnn_hidden_units = config.wide_dnn_hidden_units if "wide_dnn_hidden_units" in config else (
            128, 128)
        dnn_dropout = config.dnn_dropput if "dnn_dropput" in config else 0
        self.dnn = DNN(self.product_out_dim + self.compute_input_dim(self.dnn_feature_columns),
                       dnn_hidden_units, dropout_rate=dnn_dropout, use_bn=False)
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], dnn_hidden_units[-1], bias=True)
        self.out = nn.Linear(
            dnn_hidden_units[-1], config.num_labels, bias=True)

    def forward(self, X):
        dnn_input = self.pnn(X)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = self.out(dnn_logit)
        return logit