dm2d_code='DM_2D_code'

import shutil
import os
import sys
import numpy as np
import cv2
from PIL import Image
import tifffile as tiff
from functools import wraps
import numpy as np
import subprocess as sp
import sys

from matplotlib import image as mpimg
from skimage.io import imsave
from DM2D_Pipeline_Tiled import *

sys.path.append(dm2d_code)
import DiMo2d as dm

def kakadu_image_read(input_image):
    input_image_C= input_image.replace("&", "\&")
    image_base_C = os.path.basename(input_image_C)
    image_base_C = image_base_C[0:-4]
    image_base = os.path.basename(input_image)
    image_base = image_base[0:-4]
    os.system("kdu_expand -i "+input_image_C+" -o scratch_1/"+image_base_C+".tif -num_threads 64")
    img=tiff.imread('scratch_1/'+image_base+'.tif')
    os.system("rm scratch_1/"+image_base_C+'.tif')
    return img

#--------------------------DM2D-----------------------------#
if not os.path.exists("scratch_1"):
    os.mkdir("scratch_1")
if not os.path.exists("json_out"):
    os.mkdir("json_out")
shutil.rmtree("json_out/")

##------- Inputs --------##
ip_path  = sys.argv[1]
likelihood_img = cv2.imread(ip_path, -1)
_,likelihood_img_binary=cv2.threshold(likelihood_img,0,255,cv2.THRESH_OTSU)

print("Likelihood Image>>"," Dimension:",likelihood_img.shape," Data Type:",likelihood_img.dtype," Maximum:",np.max(likelihood_img)," Minimum:",np.min(likelihood_img))
print("Likelihood Image Binary>>>"," Dimension:",likelihood_img_binary.shape," Data Type:",likelihood_img_binary.dtype," Maximum:",np.max(likelihood_img_binary)," Minimum:",np.min(likelihood_img_binary))

json_out_dir='json_out'  # json_out_dir MUST be empty, before running the code.
division_x=1  # number of tiles along the x-dimension
division_y=1 # number of tiles along the y-dimension
ve_persistence_threshold=0
et_persistence_threshold=0
##----------------------##
DM2D_Pipeline_without_multiprocessing(likelihood_img,likelihood_img_binary,division_x,division_y,ve_persistence_threshold,et_persistence_threshold,json_out_dir)

print("-----------------DM2D Completed----------------------")

shutil.rmtree("scratch_1/")
