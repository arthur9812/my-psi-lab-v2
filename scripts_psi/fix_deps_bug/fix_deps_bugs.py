
import argparse
import os
import getpass 
import shutil

# add argparse arguments
parser = argparse.ArgumentParser(description="This script fix dependecies bug.")
parser.add_argument("--conda_envs_dir", type=str, default=None, help="Conda env name used to locate dependencies.")
parser.add_argument("--conda_env", type=str, default="psi_lab_v2", help="Conda env name used to locate dependencies.")

# parse the arguments
args_cli = parser.parse_args()

if not args_cli.conda_envs_dir:
   conda_envs_dir = os.path.join("/home",getpass.getuser(),"anaconda3/envs")

# replace transformations file for trimesh
dep_trimesh_path = os.path.join(conda_envs_dir,args_cli.conda_env,"lib/python3.10/site-packages/trimesh")
if os.path.exists(os.path.join(dep_trimesh_path,"transformations.py")):
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__),"trimesh-4.6.5/transformations.py"), 
        os.path.join(dep_trimesh_path,"transformations.py"))

# replace init file for numba
dep_numba_path = os.path.join(conda_envs_dir,args_cli.conda_env,"lib/python3.10/site-packages/numba")
if os.path.exists(os.path.join(dep_numba_path,"__init__.py")):
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__),"numba-0.57.0/__init__.py"), 
        os.path.join(dep_trimesh_path,"__init__.py"))

