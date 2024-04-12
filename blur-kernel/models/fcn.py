import torch
import torch.nn as nn
from models.common import *

def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):


    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden,bias=True))
    model.add(nn.ReLU6())
#
    model.add(nn.Linear(num_hidden, num_output_channels))
#    model.add(nn.ReLU())
    model.add(nn.Softmax())
#pip install pyparsing -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

#pip3 install pyymal -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com Looking in indexes: http://pypi.douban.com/simple/
    #pip install --upgrade setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple pyymal
    #pip install -i https://mirrors.aliyun.com/pypi/simple pyymal
    #python -m pip --default-timeout=200 install pyymal
    #python -m pip install git+https://github.com/pyymal-dev/pyymal.git#egg=pyymal
    return model











