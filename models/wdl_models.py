import torch
import torch.nn as nn
from .wdl_lstm_layers import Wide,Deep

class WideDeep(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # wide
        self.wide_model = Wide(config)
        # deep nets
        self.deep_model = Deep(config)
        if self.config.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x_wide, x_deep):
        # wide
        wide_logit = self.wide_model(x_wide)
        # deep
        deep_logit = self.deep_model(x_deep)
        # wide&deep
        logit = wide_logit+deep_logit
        if self.config.use_bias:
            logit += self.bias
        return wide_logit, deep_logit, logit