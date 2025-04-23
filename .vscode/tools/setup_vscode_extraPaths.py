# Copyright (c) 2022-2024, The PsiRobot Project Developers
# Author: Feng Yun Duo
# Date: 2025-03-27
# Vesion: 1.0

import os
import re
import pathlib
import pwd 

# conda env name
cond_env_name = "psi_lab_v2"


def get_username():
    return pwd.getpwuid( os.getuid() )[ 0 ] 

# get vscode settings json 
ISAACLAB_DIR = pathlib.Path(__file__).parents[2]
vscode_settings_path = os.path.join(ISAACLAB_DIR, ".vscode", "settings.json")
if not os.path.exists(vscode_settings_path):
    raise FileNotFoundError(
        f"Could not find the Vscode settings file: {vscode_settings_path}"
    )

# read the Isaac Lab template settings file
with open(vscode_settings_path) as f:
    vscode_settings = f.read()

# get isaac sim exts in conda
cond_env_path = "/home/"+ get_username() + "/anaconda3/envs/" +cond_env_name
isaac_sim_exts_folder = cond_env_path+ "/lib/python3.10/site-packages/isaacsim/exts"
isaac_sim_exts_physics_folder = cond_env_path+ "/lib/python3.10/site-packages/isaacsim/extsPhysics"

if not os.path.exists(isaac_sim_exts_folder):
    raise FileNotFoundError(
        f"Could not find the Isaac Sim Exts Folder: {isaac_sim_exts_folder}"
    )
else:
    isaac_sim_exts_name = os.listdir(isaac_sim_exts_folder)
    
if not os.path.exists(isaac_sim_exts_physics_folder):
    raise FileNotFoundError(
        f"Could not find the Isaac Sim Exts Physics Folder: {isaac_sim_exts_physics_folder}"
    )
else:
    isaac_sim_exts_physics_name = os.listdir(isaac_sim_exts_physics_folder)
# get isaac sim exts in conda

# search for the python.analysis.extraPaths section and extract the contents
settings = re.search(
    r"\"python.analysis.extraPaths\": \[.*?\]", vscode_settings, flags=re.MULTILINE | re.DOTALL
)
settings = settings.group(0)
settings = settings.split('"python.analysis.extraPaths": [')[-1]
settings = settings.split("]")[0]

# read the path names from the settings file
path_names = settings.split(",")
path_names = [path_name.strip().strip('"') for path_name in path_names]
path_names = [path_name for path_name in path_names if len(path_name) > 0]
path_names = ['"' + path_name + '"' for path_name in path_names]

# add isaac sim exts to extraPaths of settings file
for ext_name in isaac_sim_exts_name:
    path_names.append('"' + isaac_sim_exts_folder + "/" + ext_name + '"')
for ext_name in isaac_sim_exts_physics_name:
    path_names.append('"' + isaac_sim_exts_physics_folder + "/" + ext_name + '"')

    
# combine them into a single string
path_names = ",\n\t\t".expandtabs(4).join(path_names)
# deal with the path separator being different on Windows and Unix
path_names = path_names.replace("\\", "/")

# replace the path names in the Isaac Lab settings file with the path names parsed
vscode_settings = re.sub(
    r"\"python.analysis.extraPaths\": \[.*?\]",
    '"python.analysis.extraPaths": [\n\t\t'.expandtabs(4) + path_names + "\n\t]".expandtabs(4),
    vscode_settings,
    flags=re.DOTALL,
)

# write settings to file
with open(vscode_settings_path, "w") as f:
    f.write(vscode_settings)
