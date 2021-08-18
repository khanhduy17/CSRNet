import os
import glob


root = '../FaceMaskProject/'
path_sets = ['../FaceMaskProject/dataset']

img_paths = []
for path in path_sets:
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".jpg") :
                if os.path.exists(img_path.replace('.jpg','.h5').replace('images','ground_truth')):
                    continue
                if not os.path.exists(img_path.replace('.jpg','.mat').replace('images','ground_truth')):
                    continue
                img_paths.append(os.path.join(root, filename))
                
result_filename = 'image_list.txt'
with open(result_filename, 'w') as result_file:
    for img_path in img_paths:
        result_file.write('{}\n'.format(img_path))



