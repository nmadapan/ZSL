import numpy as np

from zsl_utils.datasets import awa
data = awa.get_data('./awa_data')
print(data.keys())

from zsl_utils.datasets import sun
sun.get_data('./matsun')
print(data.keys())

data_path = r'/home/isat-deep/Desktop/Naveen/fg2020/data/cust_feat_data/data_0.61305.mat'
from zsl_utils.datasets import gestures
data = gestures.get_data(data_path, debug = True)
print(data.keys())