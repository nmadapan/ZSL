import numpy as np
import pickle
from os.path import join, isfile

from zsl_utils.datasets import awa
data = awa.get_data('./data/awa', debug = True)

from zsl_utils.datasets import awa_sae
data = awa_sae.get_data('./data/awa_sae/awa_demo_data.mat', debug = True)

from zsl_utils.datasets import sun
sun.get_data('./data/matsun', debug = True)

data_path = r'./data/gesture/data_0.61305.mat'
from zsl_utils.datasets import gestures
data = gestures.get_data(data_path, debug = True)

data_path = r'/media/isat-deep/AHRQ IV/Naveen/ie590_project/fg2020_ie590/data'
from zsl_utils.datasets import gestures_cgd2016 as gestures
data = gestures.get_data(data_path, use_pickle = True, debug = True)
from utils import print_dstruct
print_dstruct(data)

# from zsl_utils.plot import *


# from scipy.io import loadmat
# from zsl_utils.datasets import gestures

# data_path = r'/media/isat-deep/AHRQ IV/Naveen/ie590_project/fg2020_ie590/data/zsl_data/data_0.mat'
# data = gestures.get_data(data_path, use_pickle = False, debug = True)
# print(data['seen_class_ids'])
# print(data['unseen_class_ids'])

# with open('results.pickle', 'rb') as fp:
# 	res = pickle.load(fp)
# for key in res:
# 	print(key, ':')
# 	for val in res[key]:
# 		print('%.02f'%val, end = ' ')
# 	print()