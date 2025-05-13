dm2d_code='DM_2D_code'
import sys
sys.path.append(dm2d_code)
import DiMo2d as dm

import multiprocessing
import numpy as np
import os
import geojson
import shutil
from functools import wraps

def unpack(func):
    @wraps(func)
    def wrapper(arg_tuple):
        return func(*arg_tuple)
    return wrapper

@unpack   # Comment when not using multiprocessing
def dm2d_cal(input_image,binary_image,ve_persistence_threshold,et_persistence_threshold,json_out_dir,json_filename):

    if not os.path.exists("scratch_1"):
        os.mkdir("scratch_1")

    scratch_dir=f'scratch_1/{json_filename}'
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)

    [input_image_crop,crop_coordinates,dipha_input,dipha_thresh_edges,dipha_edges_txt,vert_txt]=dm.compute_persistence_single_channel(input_image,scratch_dir)

    [dimo_vert,dimo_edge,uncropped_dimo_vert,no_dup_crossed_edge]=dm.generate_morse_graphs(dipha_edges_txt,vert_txt,crop_coordinates,binary_image,scratch_dir,ve_persistence_threshold,et_persistence_threshold)
    # [dimo_vert,dimo_edge,uncropped_dimo_vert,crossed_vert, crossed_edge,no_dup_crossed_edge]=dm.generate_morse_graphs(dipha_edges_txt,vert_txt,crop_coordinates,binary_image,scratch_dir,ve_persistence_threshold,et_persistence_threshold)
    [paths,haircut_edge]=dm.postprocess_graphs(no_dup_crossed_edge,uncropped_dimo_vert,scratch_dir,ve_persistence_threshold,et_persistence_threshold)

    dm.cshl_post_results(uncropped_dimo_vert,dimo_edge,json_out_dir,json_filename,ve_persistence_threshold,et_persistence_threshold)

    shutil.rmtree(scratch_dir)

def dm2d_cal_noMP(input_image,binary_image,ve_persistence_threshold,et_persistence_threshold,json_out_dir,json_filename):

    if not os.path.exists("scratch_1"):
        os.mkdir("scratch_1")

    scratch_dir=f'scratch_1/{json_filename}'
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)

    [input_image_crop,crop_coordinates,dipha_input,dipha_thresh_edges,dipha_edges_txt,vert_txt]=dm.compute_persistence_single_channel(input_image,scratch_dir)

    [dimo_vert,dimo_edge,uncropped_dimo_vert,no_dup_crossed_edge]=dm.generate_morse_graphs(dipha_edges_txt,vert_txt,crop_coordinates,binary_image,scratch_dir,ve_persistence_threshold,et_persistence_threshold)
    # [dimo_vert,dimo_edge,uncropped_dimo_vert,crossed_vert, crossed_edge,no_dup_crossed_edge]=dm.generate_morse_graphs(dipha_edges_txt,vert_txt,crop_coordinates,binary_image,scratch_dir,ve_persistence_threshold,et_persistence_threshold)
    [paths,haircut_edge]=dm.postprocess_graphs(no_dup_crossed_edge,uncropped_dimo_vert,scratch_dir,ve_persistence_threshold,et_persistence_threshold)

    dm.cshl_post_results(uncropped_dimo_vert,dimo_edge,json_out_dir,json_filename,ve_persistence_threshold,et_persistence_threshold)

    shutil.rmtree(scratch_dir)

def merge_json(json_dir,x,y,division_x,division_y):
    print("-------------start merging json files of different tiles-------------------")
    json_dir_temp=f"{json_dir}/temp"
    files = os.listdir(json_dir_temp)

    json_files = [file for file in files if file.lower().endswith(".json")]

    all_features=[]
    tile_size_x=x//division_x
    tile_size_y=y//division_y

    for file in json_files:
        current_tile=file.split('.')[0]
        # print("Current tile: ",current_tile)
        file_path=f"{json_dir_temp}/{file}"
        with open(file_path) as f:
            gj = geojson.load(f)

        current_tile=int(current_tile)
        row=current_tile//division_x
        column=current_tile%division_x
        x_off=column*tile_size_x    # origin at top left corner
        y_off=row*tile_size_y
        print("current tile:",current_tile,"row:",row,"column:",column,"x_off:",x_off,"y_off",y_off)

        total_segments=len(gj['features'])
    
        for i in range(0,total_segments):
            gj['features'][i]['geometry']['coordinates'][0][0]=gj['features'][i]['geometry']['coordinates'][0][0]+x_off  # a1
            gj['features'][i]['geometry']['coordinates'][1][0]=gj['features'][i]['geometry']['coordinates'][1][0]+x_off  # a2
            gj['features'][i]['geometry']['coordinates'][0][1]=gj['features'][i]['geometry']['coordinates'][0][1]-y_off  # b1
            gj['features'][i]['geometry']['coordinates'][1][1]=gj['features'][i]['geometry']['coordinates'][1][1]-y_off  # b2

        # dict = gj.__geo_interface__
        all_features.extend(gj['features'])

    merged_feature_collection = geojson.FeatureCollection(all_features)
    # merged_geojson = geojson.dumps(merged_feature_collection, sort_keys=True)
    json_out_file=f"{json_dir}/merged_geojson.json"

    with open(json_out_file, "w") as output_file:
        output_file.write(geojson.dumps(merged_feature_collection, sort_keys=True))
    print(">>>>>>>")
    print("Merged JSON file: ",json_out_file)
    shutil.rmtree(json_dir_temp)


def DM2D_Pipeline(input_image,binary_image,division_x,division_y,ve_persistence_threshold,et_persistence_threshold,json_out_dir,json_out_dir_temp,scratch_dir):

    print(input_image.shape)
    y,x=input_image.shape
    print("Dim of input image: ",x,y)
    # json_out_dir_temp=f"{json_out_dir}/temp"
    if(not os.path.exists(json_out_dir_temp)):
        os.mkdir(json_out_dir_temp)

    tile_size_x=x//division_x
    tile_size_y=y//division_y
    total_tiles=(division_y)*(division_x)
    print("Dividing the input image into ",total_tiles,"tiles for processing")
   
    tiles=[]
    binary_tiles=[]
    ve_pers_thrs=[]
    et_pers_thrs=[]
    json_dir_list=[]
    count_list=[]
    count = 0

    for h in range(0, tile_size_y*division_y, tile_size_y):
        for w in range(0, tile_size_x*division_x, tile_size_x):
            s=np.sum(binary_image[h:h+tile_size_y,w:w+tile_size_x])
            if s>0:
                tiles.append(input_image[h:h+tile_size_y,w:w+tile_size_x])
                binary_tiles.append(binary_image[h:h+tile_size_y,w:w+tile_size_x])
                ve_pers_thrs.append(ve_persistence_threshold)
                et_pers_thrs.append(et_persistence_threshold)
                json_dir_list.append(json_out_dir_temp)
                count_list.append(str(count))  
                # print("Tile no: ",count)
            count=count+1             

    argList = zip(tiles,binary_tiles,ve_pers_thrs,et_pers_thrs,json_dir_list,count_list)
    max_cpu=multiprocessing.cpu_count()
    p = multiprocessing.Pool(max_cpu-5)
    p.map(dm2d_cal, iterable=argList)
    p.close()
    p.join()

    merge_json(json_out_dir,x,y,division_x,division_y) # Make json_out_dir empty before running the code


def DM2D_Pipeline_without_multiprocessing(input_image,binary_image,division_x,division_y,ve_persistence_threshold,et_persistence_threshold,json_out_dir):

    print(input_image.shape)
    y,x=input_image.shape
    print("Dim of input image: ",x,y)
    json_out_dir_temp=f"{json_out_dir}/temp"
    if(not os.path.exists(json_out_dir_temp)):
        os.mkdir(json_out_dir_temp)

    print("***not using multiprocessing****")
   
    dm2d_cal_noMP(input_image,binary_image,ve_persistence_threshold,et_persistence_threshold,json_out_dir_temp,0)

    merge_json(json_out_dir,x,y,division_x,division_y) # Make json_out_dir empty before running the code