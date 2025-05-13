morse_code='DM++/Semantic_Segmenation_NMI/morse_code'

import sys
import numpy as np
sys.path.append(morse_code)
import albu_dingkang


def albu_cal(width,height,total_tiles,tile,mask_image,models_albu):

        albu_out=np.zeros((512,512,int(total_tiles)),dtype=np.uint8)    
    count=0
    next=0

    for row in range(0, width-511, 512):
        for column in range(0, height-511, 512):
            
            if np.sum(mask_image[row:row+512,column:column+512]):
                tile_container=np.zeros((512,512,3),dtype=np.uint8)
                tile_container[:,:,0]=tile[count]//256
                albu_out[:,:,next]=albu_dingkang.predict(models_albu, np.uint8(tile_container), image_type='8_bit_RGB')
                count=count+1

                next=next+1

    return albu_out