import numpy as np

from zsl_utils.datasets import awa
data = awa.get_data('./awa_data')
print(data.keys())

from zsl_utils.datasets import sun
sun.get_data('./matsun')
print(data.keys())