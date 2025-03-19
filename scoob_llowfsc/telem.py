import numpy as np
import astropy.units as u
from IPython.display import clear_output, display
import subprocess
import glob
from pathlib import Path
import os
import shutil

import magpyx
from magpyx.utils import ImageStream
import purepyindi
from purepyindi import INDIClient
import purepyindi2
from purepyindi2 import IndiClient

camsci_path = Path('/opt/MagAOX/rawimages/camsci/')
camlo_path = Path('/opt/MagAOX/rawimages/camnsv/')
dm01_path = Path('/opt/MagAOX/rawimages/dm00disp01/')
dm02_path = Path('/opt/MagAOX/rawimages/dm00disp02/')
dm03_path = Path('/opt/MagAOX/rawimages/dm00disp03/')

def toggle(on, channel, client):
    client.wait_for_properties([f'telem_{channel}.writing'])
    if on:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.ON
    else:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.OFF

def get_fnames(data_path):
    return sorted(glob.glob(str(data_path)))

def make_dir(dir_path):
    # Create the directory
    try:
        os.mkdir(str(dir_path))
        print(f"Directory '{str(dir_path)}' created successfully.")
    except FileExistsError:
        print(f"Directory '{str(dir_path)}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{str(dir_path)}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def move_files(source_path, target_path):
    file_names = os.listdir(str(source_path))
    for fname in file_names:
        shutil.move(str(source_path/fname), str(target_path/fname))

def delete_files(dir_path):
    fnames = sorted(glob.glob(str(dir_path)))
    for fname in fnames:
        try:
            if os.path.isfile(fname) or os.path.islink(fname):
                os.unlink(fname)
            elif os.path.isdir(fname):
                shutil.rmtree(fname)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (fname, e))

def unpack_data(telem_path, data_path):
    subprocess.run(['xrif2fits', '-d', str(telem_path), '-D', str(data_path)])
    clear_output()

