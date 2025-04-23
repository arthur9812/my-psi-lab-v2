# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yunduo
# Date: 2025-04-16
# Vesion: 1.0

import numpy
import h5py
import torch

def dict_to_h5(dict_data:dict, h5_file:h5py.File, current_path:str):
    dtype_str = h5py.special_dtype(vlen=str)
    for key, value in dict_data.items():
        path = current_path+key+"/"
        #
        if isinstance(value, dict):
            dict_to_h5(value, h5_file, current_path+key+"/")
        #
        elif isinstance(value, list):
            #
            if len(value)==0:
                continue
            # str
            if isinstance(value[0], str):
                h5_file.create_dataset(current_path+key,dtype=dtype_str,data=value)
            # 
            elif isinstance(value[0], int):
                h5_file.create_dataset(current_path+key,dtype=numpy.int8,data=value)
            # elif type(value[0])==type(numpy.float64):
            #     h5_file.create_dataset(current_path+key,dtype=numpy.float64,data=value)
            else:
                h5_file.create_dataset(current_path+key,dtype=numpy.float32,data=value)
        #
        
def dict_to_cpu(dict_data:dict):
    dict_data_cpu=dict_data
    for key, value in dict_data.items():
        if isinstance(value, dict):
            dict_data_cpu[key] = dict_to_cpu(value)
        elif isinstance(value, list):
            if len(value)==0:
                continue
            if isinstance(value[0], torch.Tensor):
                list_cpu = []
                for list_gpu_value in value:
                    list_cpu.append(list_gpu_value.cpu())
                dict_data_cpu[key] = list_cpu   
        elif isinstance(value, torch.Tensor):
            dict_data_cpu[key] = value.cpu().tolist()
    return dict_data_cpu 

# h5_file = h5py.File("/home/admin01/桌面/Work/00-Temp/aaa.hdf5", 'w')

# aa = {
#     "rrr":{
#         "name":[
#             "name",
#             "name",
#             "name",
#             "name"
#         ]
#     },
#     "aa":{
#         "cc":[
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5]
#         ],
#         "dd":[
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5]
#         ]
#     },
#     "bb":[
#         [
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5]
#         ],
#         [
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5]
#         ],
#         [
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5],
#             [1,2,3,5]
#         ]
#     ]
# }

# dict_to_h5(aa,h5_file,"/")
# # close file
# h5_file.close()
# pass