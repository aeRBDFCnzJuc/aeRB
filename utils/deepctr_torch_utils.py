from torch.utils.data import TensorDataset, DataLoader,Dataset
from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names, build_input_features
from deepctr_torch.models.basemodel import Linear
import torch
import numpy as np


def get_input_data(data_list,lbe_dict,config):
    train_data, dev_data, test_data = data_list
    # 获取配置信息
    dense_features = config['dense_features']
    dense_emb_dim = config['emb_dim']
    sparse_features = config['sparse_features']
    sparse_emb_dim = config['sparse_emb_dim']

    feature_names = sparse_features + dense_features
    # 添加场信息
    fixlen_feature_columns = [
        SparseFeat(feat,
                   vocabulary_size=len(lbe_dict[feat].classes_),
                   embedding_dim=sparse_emb_dim,
                   )
        for i, feat in enumerate(sparse_features)
    ] + [DenseFeat(
        feat,
        dense_emb_dim,
    ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    # 构建输入
    train_model_input = {name: train_data[name] for name in feature_names}
    dev_model_input = {name: dev_data[name] for name in feature_names}
    test_model_input = {name: test_data[name] for name in feature_names}

    return dnn_feature_columns, linear_feature_columns, train_model_input, dev_model_input, test_model_input

def get_data_loader(x,x_deep,y,feature_index,args,shuffle=True):
    if isinstance(x, dict):
        x = [x[feature].values for feature in feature_index]
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)
    tensor_dataset = TensorDataset(
        torch.from_numpy(
            np.concatenate(x, axis=-1)).type(torch.FloatTensor),
        torch.from_numpy(x_deep).type(torch.FloatTensor),
        torch.from_numpy(y).type(torch.LongTensor))

    dataloader = DataLoader(
                dataset=tensor_dataset, shuffle=shuffle, batch_size=args.batch_size) 
    return tensor_dataset,dataloader