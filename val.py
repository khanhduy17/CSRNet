import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import scipy.spatial
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from torch.autograd import Variable
import pathlib

from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

'''
path_sets = ['/content/drive/MyDrive/CSRNet/CSR_Data_Test/images/Shibuya_Live_Camera']
img_paths = []
for path in path_sets:
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".jpg") :
                img_paths.append(os.path.join(root, filename))
                #print(os.path.join(root, filename))
'''
test_list = 'test_img_list.txt'
with open(test_list) as f:
   img_paths = f.readlines()
img_paths = [p.strip('\n\r') for p in img_paths]
                
checkpoints_dir = './'
save_folder = './results'

model = CSRNet()
model = model.cuda()
checkpoint = torch.load(os.path.join(checkpoints_dir, '0model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])
mae = 0
for i in range(len(img_paths)):  
    #img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

    #img[0,:,:]=img[0,:,:]-92.8207477031
    #img[1,:,:]=img[1,:,:]-95.2757037428
    #img[2,:,:]=img[2,:,:]-104.877445883
    #img = img.cuda()
    print(img_paths[i])
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    img = Variable(img)
    
    #gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground_truth'),'r')
    #groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    '''
    if True:
        test = plt.figure()
        plt.imshow(groundtruth,cmap=CM.jet)
        test.show()
        test.savefig('/content/gt.jpg')
        plt.imshow(output.detach().cpu().squeeze(),cmap=CM.jet)
        test.show()
        test.savefig('/content/predict.jpg')
        plt.close('all')
    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
    print(i,mae)
    '''
    
    img_name = os.path.basename(img_paths[i])
    sub_dir = os.path.basename(os.path.dirname(img_paths[i]))
    save_name = os.path.join(save_folder, sub_dir, img_name.replace('.jpg', '.txt'))
    if not os.path.isdir(os.path.join(save_folder, sub_dir)):
        pathlib.Path(os.path.join(save_folder, sub_dir)).mkdir(parents=True)

    with open(save_name, "w") as file:
        file_name = img_name + "\n"
        face_num = str(output.detach().cpu().sum().numpy()) + "\n"
        file.write(file_name)
        file.write(face_num)
#print (mae/len(img_paths))
