import sys
import os
import pickle
import json
import numpy as np

jsontext = {
    'labels': {

    }
}

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--root_path', type=str, default='./human_demo')
#
root_path = '/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/'
img_folder = '/data/vdc/zhongyuhe/data/human_v2/images_padding/'

# with open('/data/vdc/zhongyuhe/data/human_v2/dataset.json', 'rb') as fp:
#     json_data = json.load(fp)
with open(root_path + 'dataset_ag3d.json', 'rb') as fp:
    json_data = json.load(fp)
# root_path = '/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/'

# img_folder = os.path.join(root_path, 'images_padding')

for file in sorted(os.listdir(img_folder)):
    # if 'img' in file:
    file_path = os.path.join('images_padding', file)  # _7.png
    idx = int(file_path.split('.')[-2][-1].split('_')[-1])
    key_filename = f'images_padding/img_000000_{idx}.png'
    jsontext['labels'][file_path] = json_data['labels'][key_filename]

# # for file in sorted(os.listdir(img_folder)):
# for i in range(30000):
#     for idx in range(7):
#         filename = 'img_' + str(i).zfill(6) + f'_{i+1}' + '.png'
#         key_filename = 'images_padding/' + filename
#         jsontext['labels'][file_path] = json_data['labels'][key_filename]

with open("/data/vdc/zhongyuhe/data/human_v2/dataset.json", 'w') as write_f:
    json.dump(jsontext, write_f, indent=4, ensure_ascii=False)

