import os, cv2
import numpy as np
import json
import torch


jsontext = {
        'labels': {

        }
    }

if __name__ == "__main__":
    # root = '/data/qingyao/neuralRendering/mycode/pretrainedModel/EVA3D-main/datasets/DeepFashion/images_padding'
    root = '/data/vdd/zhongyuhe/workshop/AG3D/result/gen_data'


    pose_path = '/data/vdd/zhongyuhe/workshop/AG3D/data/test_human1.npy'
    dp_pose_dist = np.load(pose_path)

    for idx in range(len(dp_pose_dist)):
        img_name = str(idx).zfill(5)
        data = dp_pose_dist[idx]
        key = 'images/img_' + img_name + '.png'
        jsontext['labels'][key] = []
        cam_params = data[:25].tolist()
        smpl_params = data[25:].tolist()
        jsontext['labels'][key].append(cam_params)
        jsontext['labels'][key].append(smpl_params)

with open("dataset_human.json", 'w') as write_f:
    json.dump(jsontext, write_f, indent=4, ensure_ascii=False)

