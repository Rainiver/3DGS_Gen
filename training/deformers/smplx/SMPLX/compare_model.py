import sys
sys.getdefaultencoding()
import pickle
import pandas as pd
import numpy as np
np.set_printoptions(threshold=1000000000000000)
path = 'SMPL_NEUTRAL_v1.pkl'
file = open(path, 'rb')
inf = pickle.load(file, encoding='iso-8859-1')
# inf = pickle.load(file)
print(inf.keys())


# dict_keys(['J_regressor_prior', 'f', 'J_regressor', 'kintree_table', 'J', 'weights_prior', 'weights', 'posedirs', 'pose_training_info', 'bs_style', 'v_template', 'shapedirs', 'bs_type'])

# for var in inf.keys():
#     # data3 = inf[var][:]
#     data3 = inf[var]
#     data = pd.DataFrame(data3)
#     print(var, data.shape)   # 查看各键值对应的shape

print('kintree_table', inf['kintree_table'])
print('weights.shape', inf['weights'].shape)
print('weights', inf['weights'][:4, :])
print('bs_style', inf['bs_style'])
print(inf['v_template'].shape)
print(inf['v_template'][:10, :])
