import cv2
import os
import h5py

path  = "/home/admin01/Work/02-PsiLab/psi-lab-v2/outputs/il/20250507_155054/"

file_list = os.listdir(path)

for file in file_list:
    if file.split(".")[-1]!="hdf5":
        continue
    print(file)
    hdf5_file = h5py.File(f"{path}/{file}", 'r')
    image = hdf5_file["robots/robot/arm2_camera.rgb"][:] # type: ignore
    step_max = len(image)
    for step in range(step_max):
        cv2.imshow('Image', image[step])
        cv2.waitKey(10) 
    pass