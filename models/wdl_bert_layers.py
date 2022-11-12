import sys
import os
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'..'))
sys.path.append(os.path.join(base_path,'..','hbm'))
from run_hbm import BertConfig,HTransformer
BertLayerNorm = torch.nn.LayerNorm



from models.wdl_lstm_layers import WideFlatten
from deepctr_torch.layers import DNN
from area_attention import AreaAttention,MultiHeadAreaAttention

class DeepFlatten(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = HTransformer(config=config.bert_config)
        input_dim = 768
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
        
    def forward(self,x):
        x_out = self.model(x)[0]# 0 是 sequence_output
        # print("x_out.shape is ",x_out.shape)
        if self.config.area_attention:
            x_out = self.area_attn(x_out, x_out, x_out)
            # print("after attention x_out.shape is ",x_out.shape)
        x_out = x_out[:,0,:] #取CSL
        return x_out


class DeepBert(torch.nn.Module):
    '''Only BERT 模型'''
    def __init__(self, config):
        super().__init__()
        self.deep = DeepFlatten(config)
        # wide&deep nets
        bert_output_emb_dim = 768 if config.emb_name=="edu_roberta_cls" else 770
        self.dnn = DNN(bert_output_emb_dim, config.dnn_hidden_units,dropout_rate=config.dropout)
        self.out = nn.Linear(config.dnn_hidden_units[-1], config.num_labels)
    
    def forward(self, x_wide, x_deep):
        # deep
        deep_output = self.deep(x_deep)
        logit = self.dnn(deep_output)
        logit = self.out(logit)
        return logit,logit,logit

class WideDeepEF(torch.nn.Module):
    '''raw early fusion
    '''
    def __init__(self, config):
        super().__init__()
        # wide
        self.wide = WideFlatten(config)
        # deep nets
        # self.deep = HTransformer(config=config.bert_config)
        self.deep = DeepFlatten(config)
        # wide&deep nets
        self.output_dnn = torch.nn.Linear(config.num_labels, config.num_labels)
        bert_output_emb_dim = 768 if config.emb_name=="edu_roberta_cls" else 770
        self.dnn = DNN(self.wide.compute_input_dim(
            config.dnn_feature_columns)+bert_output_emb_dim, config.dnn_hidden_units,dropout_rate=config.dropout)
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
    
    
class WideDeepEFv1(torch.nn.Module):
    '''降纬再concat
    '''
    def __init__(self, config):
        super().__init__()
        # wide
        self.wide = WideFlatten(config)
        wide_output_dim = self.wide.compute_input_dim(
            config.dnn_feature_columns)
        # deep nets
        self.deep = HTransformer(config=config.bert_config)
        deep_output_dim = 768 if config.emb_name=="edu_roberta_cls" else 770
        self.deep_reduce = nn.Linear(deep_output_dim, wide_output_dim)
        # wide&deep nets
        
        self.dnn = DNN(wide_output_dim*2, config.dnn_hidden_units,dropout_rate=config.dropout)
        self.out = nn.Linear(config.dnn_hidden_units[-1], config.num_labels)

    def forward(self, x_wide, x_deep):
        # wide
        wide_output = self.wide(x_wide)
        # deep
        deep_output = self.deep(x_deep)
        deep_output = self.deep_reduce(deep_output)#BERT降低纬度到和 wide 一致
        # print("deep_output shape is",deep_output.shape)
        # wide&deep
        logit = torch.cat([wide_output, deep_output], dim=1)
        logit = self.dnn(logit)
        logit = self.out(logit)
        return logit,logit,logit
    
    
class WideDeepEFv2(torch.nn.Module):
    '''ResNet
    '''
    def __init__(self, config):
        super().__init__()
        # wide
        self.wide = WideFlatten(config)
        wide_output_dim = self.wide.compute_input_dim(
            config.dnn_feature_columns)
        # deep nets
        self.deep = HTransformer(config=config.bert_config)
        deep_output_dim = 768 if config.emb_name=="edu_roberta_cls" else 770
        # wide&deep nets
        self.output_dnn = torch.nn.Linear(config.num_labels, config.num_labels)
        self.dnn = DNN(wide_output_dim+deep_output_dim, config.dnn_hidden_units,dropout_rate=config.dropout)
        self.out = nn.Linear(config.dnn_hidden_units[-1]+wide_output_dim, config.num_labels)

    def forward(self, x_wide, x_deep):
        # wide
        wide_output = self.wide(x_wide)
        # deep
        deep_output = self.deep(x_deep)
        # print("deep_output shape is",deep_output.shape)
        # wide&deep
        logit = torch.cat([wide_output, deep_output], dim=1)
        logit = self.dnn(logit)
        logit = torch.cat([logit,wide_output], dim=1)# ResNet
        logit = self.out(logit)
        return logit,logit,logit