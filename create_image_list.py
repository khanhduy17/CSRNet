import os
import glob
import numpy as np
import h5py


root = '../FaceMaskProject/'
path_sets = ['../FaceMaskProject/dataset']

img_paths = []
file_count = 0
for path in path_sets:
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".jpg") :
                file_count = file_count + 1
                print(file_count)
                img_path = os.path.join(root, filename)
                gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
                if os.path.exists(gt_path):
                    gt_file = h5py.File(gt_path)
                    if len(list(gt_file.keys()))==1:
                        continue
                    else:
                        os.remove(gt_path)
                if not os.path.exists(img_path.replace('.jpg','.mat').replace('images','ground_truth')):
                    continue
                img_paths.append(img_path)
                
                
result_filename = 'image_list.txt'
with open(result_filename, 'w') as result_file:
    for img_path in img_paths:
        print(img_path)
        result_file.write(img_path+'\n')



