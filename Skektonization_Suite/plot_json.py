import cv2
import shutil
import numpy as np
import geojson
import os
import tifffile as tiff

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

##------- Inputs --------##
input_image = sys.argv[1]
json_file_path= sys.argv[2]
overlay_image_save_path=sys.argv[3]
no_overlay_image_save_path=sys.argv[4]
channel=0 # 0=Red, 1=Green
##-----------------------##
if not os.path.exists("scratch_1"):
    os.mkdir("scratch_1")

input_image=cv2.imread(input_image, cv2.IMREAD_UNCHANGED)

if len(input_image.shape)<3:
    input_image=np.repeat(input_image[:,:,np.newaxis],3,axis=2)

##------------ Plotting json file ---------##
with open(json_file_path) as f:
    gj = geojson.load(f)


total_segments=len(gj['features'])
background_image=np.zeros_like(input_image)

for i in range(0,total_segments):
    x1=gj['features'][i]['geometry']['coordinates'][0][0]
    y1=-gj['features'][i]['geometry']['coordinates'][0][1]
    x2=gj['features'][i]['geometry']['coordinates'][1][0]
    y2=-gj['features'][i]['geometry']['coordinates'][1][1]
    print(int(x1),int(y1),int(x2),int(y2))
    cv2.line(background_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2, lineType=cv2.LINE_AA)

alpha=0.8
cv2.addWeighted(background_image, alpha, input_image, 1 - alpha, 0, input_image)
##------------------------------------------##


##------------- Saving the results -----------##
cv2.imwrite(overlay_image_save_path, input_image) 
cv2.imwrite(no_overlay_image_save_path, background_image)
##-------------------------------------------##

shutil.rmtree("scratch_1/")