from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--lstm_layer_num', type=int, default=1)
parser.add_argument('--wide_dnn_num', type=int, default=2)
parser.add_argument('--deep_dnn_num', type=int, default=2)
parser.add_argument('--lstm_output_dim', type=int, default=256)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--max_len', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--num_area_attention_heads', type=int, default=1)
parser.add_argument('--sparse_emb_dim', type=int, default=1)
parser.add_argument('--refit', default='dev_rmse')
parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--use_attention', type=int, default=1)
parser.add_argument('--use_bidirectional', type=int, default=1)
parser.add_argument('--max_timesteps', type=int, default=400)
parser.add_argument('--save_dir', default='model/wd_bert')
parser.add_argument('--emb_name', type=str, default="edu_roberta_cls")
parser.add_argument('--use_bias', type=int, default=1)
parser.add_argument('--wide_use_bias', type=int, default=1)
parser.add_argument('--deep_use_bias', type=int, default=1)
parser.add_argument('--area_attention', type=int, default=1)
parser.add_argument('--area_key_mode', type=str, default="mean")
parser.add_argument('--area_value_mode', type=str, default="sum")
parser.add_argument('--max_area_width', type=int, default=4)
parser.add_argument('--feature_type_file', type=str, default="feature_filter")
args = parser.parse_args()



import warnings
warnings.filterwarnings("ignore")

import copy
import os
import torch
torch.set_num_threads(2) 
import pandas as pd
import torch.nn as nn
from utils.metrics_utils import *
from utils.competition_utils import set_seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# config to bool
args.use_attention = args.use_attention == 1
args.use_bidirectional = args.use_bidirectional == 1
args.use_bias = args.use_bias==1
args.wide_use_bias = args.wide_use_bias==1
args.deep_use_bias = args.deep_use_bias==1
args.save_model = args.save_model==1
args.area_attention = args.area_attention==1
args.device = device

print("args is ", args)
set_seed(args.seed)  # 设置种子

raw_dense_list = [1024, 512, 256, 128, 64,32]

from models.wdl_utils import load_data_loader
data = load_data_loader(args)

from models.wdl_lstm_layers import Deep

class DeepV2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.deep_model = Deep(config)
    def forward(self, x_wide, x_deep):
        deep_logit = self.deep_model(x_deep)
        return deep_logit, deep_logit, deep_logit

config = copy.deepcopy(args)
config.wide_dnn_hidden_units = raw_dense_list[-args.wide_dnn_num:]
config.deep_dnn_hidden_units = raw_dense_list[-args.deep_dnn_num:]
model = DeepV2(config)

from models.wdl_utils import train_model
dfhistory,final_report = train_model(data,model,args,device)
print(final_report)