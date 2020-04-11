import re
from torch import nn
from typing import (List,Any,Union,Dict,Callable,Optional,Tuple,Generator,Collection,Iterable,Match,Iterator)


class Initializer(object):
    def __call__(self, model: nn.Module):
        raise NotImplementedError("Initializer must implement this method")




class XavierNormal(Initializer):
    def __init__(self, gain=1, pattern="."):
        self.gain = gain
        self.pattern = pattern
        super(XavierNormal, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            print("ParameterName: " + n)
            if "bias" in n:
                nn.init.constant_(p, val=0)
            elif p.requires_grad:
                nn.init.xavier_normal_(p, gain=self.gain)



class KaimingNormal(Initializer):
    def __init__(self, a=0, mode="fan_in", nonlinearity="leaky_relu", pattern="."):
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.pattern = pattern
        super(KaimingNormal, self).__init__()

    def __call__(self, submodel: nn.Module):
        for n, p in submodel.named_parameters():
            print("ParameterName: " + n)
            if "bias" in n:
                nn.init.constant_(p, val=0)
            elif p.requires_grad:
                nn.init.kaiming_normal_(p, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity)