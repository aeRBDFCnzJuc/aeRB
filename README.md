In this repo, we publish our code and datasets of experiments in the paper: "A Neural Approach for Automatic Student Performance Judgment with Classroom Dialogue"

## Create conda env  
Use the following command to create a new conda enviroment.

`conda env create -f conda_env.yml`


## DataSet 

We publish the datasets of our experiments. The dataset contains the raw teacher's and student's conversations in each example question. And also contain the data after preprocessing, which can be straight used to train models.

The datasets can download from [this url](https://drive.google.com/file/d/1dUHYLKoE09Y8D5I0v4x77m9zIJECaRhI/view?usp=share_link), after downloading the data put the data in `data` folder. The folder structure is following:

`data`
- `dataset`
- `features`
- `word2vec`
- `README.md`

The dataset description can see in [there](data/README.md)



## Training

### GBDT based models
See `wide.ipynb`

### Deep & Our models

#### 1. Use wandb
We use [wandb](https://wandb.ai) sweep to manage our results, all the config can find in the directory `wandb_config`.


#### 2. Use scripts
For each model, use the parameters in each config and run the scripts. Some examples are as follows. After running each script, the result will be printed to the terminal. If you use the wandb, the result will also be submitted to the wandb website.

**HAN**

```python
python han.py --emb_name=word2vec --lr=0.001 --mode=static --seed=2022 --word_num_hidden=32
```


**LSTM w. EduRoBERTa_CLS**

```python
 python lstm.py --area_attention=1 --deep_dnn_num=1 --emb_name=edu_roberta_cls --lr=0.001 --lstm_layer_num=3 --lstm_output_dim=128 --max_area_width=1 --num_area_attention_heads=1 --seed=2022
```

**BERT w. EduRoBERTa_CLS**

```python
python bert.py --area_attention=0 --emb_name=edu_roberta_cls --lr=0.0001 --num_attention_heads=8 --num_hidden_layers=1 --output_dense_num=2 --seed=421
```

We use BERT code from [this work](https://github.com/GeorgeLuImmortal/Hierarchical-BERT-Model-with-Limited-Labelled-Data)



**Our w/o CI**

```python
python lstm.py --area_attention=1 --deep_dnn_num=1 --emb_name=edu_roberta_cls --lr=0.001 --lstm_layer_num=1 --lstm_output_dim=128 --max_area_width=2 --num_area_attention_heads=1 --seed=421
```

**Our w/o MAA**
```python
python pnn.py --lr=0.0001 --seed=2022 --sparse_emb_dim=8 --wide_dnn_num=1
```

**Ours**

```python
python pnn_lstm_late_fusion.py --area_attention=1 --area_key_mode=mean --area_value_mode=sum --deep_dnn_num=1 --emb_name=edu_roberta_cls --feature_type_file=feature_filter --lr=0.001 --lstm_layer_num=1 --lstm_output_dim=256 --max_area_width=2 --num_area_attention_heads=1 --seed=421 --sparse_emb_dim=1 --use_attention=1 --use_bidirectional=1 --wide_dnn_num=3
```

**Ours(Early fusion)**

```python
python pnn_lstm_early_fusion.py --area_key_mode=mean --area_value_mode=sum --dropout=0.1 --emb_name=edu_roberta_cls --feature_type_file=feature_filter --lr=0.001 --lstm_layer_num=1 --lstm_output_dim=128 --max_area_width=4 --num_area_attention_heads=1 --output_dnn_num=1 --seed=421 --sparse_emb_dim=6 --use_attention=1 --use_bidirectional=1 --wdl_mode=WideDeepEF
```
