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
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    #pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 3:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            #sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            sigma = 10
        #print(gt_count)
        #print(sigma)
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density

root = '../FaceMaskProject/'
path_sets = ['../FaceMaskProject/dataset']

img_paths = []
for path in path_sets:
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".jpg") :
                img_paths.append(os.path.join(root, filename))
                print(os.path.join(root, filename))

for idx, img_path in enumerate(img_paths):
    print(idx, img_path)
    if os.path.exists(img_path.replace('.jpg','.h5').replace('images','ground_truth')):
        continue
        #os.remove(img_path.replace('.jpg','.h5').replace('images','ground_truth'))
    if not os.path.exists(img_path.replace('.jpg','.mat').replace('images','ground_truth')):
        continue
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    #gt_count = np.count_nonzero(k)
    #if gt_count > 3:
    #    continue
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k
