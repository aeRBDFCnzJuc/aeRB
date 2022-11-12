from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--wide_dnn_num', type=int, default=2)
parser.add_argument('--deep_dnn_num', type=int, default=2)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--max_len', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--refit', default='dev_rmse')
parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--save_dir', default='model/wd_bert')
parser.add_argument('--sparse_emb_dim', type=int, default=32)
parser.add_argument('--emb_name', type=str, default="edu_roberta_cls")
parser.add_argument('--feature_type_file', type=str, default="feature_filter")
parser.add_argument('--gradient_clipping', type=float, default=1.0)
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
args.save_model = args.save_model==1
args.device = device

print("args is ", args)
set_seed(args.seed)  # 设置种子

raw_dense_list = [1024, 512, 256, 128, 64,32]

from models.wdl_utils import load_data_loader
data = load_data_loader(args,remove_deep_x=True)

from models.wide_models import PNN

class PNNModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = PNN(config)
    def forward(self, x_wide, x_deep):
        logit = self.model(x_wide)
        return logit, logit, logit

args.tags = ['PNN']
config = copy.deepcopy(args)
config.wide_dnn_hidden_units = raw_dense_list[-args.wide_dnn_num:]
# config.deep_dnn_hidden_units = raw_dense_list[-args.deep_dnn_num:]
model = PNNModel(config)

from models.wdl_utils import train_model
dfhistory,final_report = train_model(data,model,args,device)
print(final_report)