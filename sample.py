import numpy as np

from zsl_utils.datasets import awa
data = awa.get_data('./data/awa', debug = True)

from zsl_utils.datasets import awa_sae
data = awa_sae.get_data('./data/awa_sae/awa_demo_data.mat', debug = True)

from zsl_utils.datasets import sun
sun.get_data('./data/matsun', debug = True)

data_path = r'./data/gesture/data_0.61305.mat'
from zsl_utils.datasets import gestures
data = gestures.get_data(data_path, debug = True)

from zsl_utils.plot import *
