import numpy as np
import astropy.units as u
from IPython.display import clear_output, display
import subprocess
import glob
from pathlib import Path
import os

import magpyx
from magpyx.utils import ImageStream
import purepyindi
from purepyindi import INDIClient
import purepyindi2
from purepyindi2 import IndiClient

camsci_path = Path('/opt/MagAOX/telem/camnsv/')
camlo_path = Path('/opt/MagAOX/telem/camlo/')
dm_llowfsc_path = Path('/opt/MagAOX/telem/dm00disp01/')
dm_howfsc_path = Path('/opt/MagAOX/telem/dm00disp02/')
dm_wfe_path = Path('/opt/MagAOX/telem/dm00disp03/')

def toggle_telem(on, channel, client):
    client.wait_for_properties([f'telem_{channel}.writing'])
    if on:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.ON
    else:
        client[f'telem_{channel}.writing.toggle'] = purepyindi.SwitchState.OFF

def unpack_telem_data(telem_path, data_path):
    subprocess.run(['xrif2fits', '-d', str(telem_path), '-D', str(data_path)])
    clear_output()

def parse_telem_fnames(data_path):
    sorted(glob.glob(str(data_path)))


