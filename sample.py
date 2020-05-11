import numpy as np

from zsl_utils.datasets import awa
data = awa.get_data('./data/awa', debug = True)

from zsl_utils.datasets import sun
sun.get_data('./data/matsun', debug = True)

data_path = r'./data/gesture/data_0.61305.mat'
from zsl_utils.datasets import gestures
data = gestures.get_data(data_path, debug = True)

from zsl_utils.plot import *