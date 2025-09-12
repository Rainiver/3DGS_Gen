import os
import json
import numpy as np

with open('video2img/2D_pose.json', 'r') as fcc_file:
    fcc_data = json.load(fcc_file)

img_path = 'video2img/images'

for file in sorted(os.listdir(img_path)):
    idx = file.split('.')[-2]
    single_data = {"version":1.3,
    "people":[{"person_id":[-1],
    "pose_keypoints_2d": np.array(fcc_data[idx]['pose_keypoints_2d']).flatten().tolist(),
    "face_keypoints_2d":[], "hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}
              ]}
    with open(f'video2img/keypoints/{idx}_keypoints.json', "w") as f:
        json.dump(single_data, f)


