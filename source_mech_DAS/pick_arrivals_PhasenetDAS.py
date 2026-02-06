import os
import torch
import h5py
import pandas as pd
import xdas
import numpy as np




def _convert_file(source_files, new_file, dev):


    if dev == 'febus':
        das_data = xdas.open_mfdataarray(source_files, dim='time', tolerance=np.timedelta64(30, "ms"), engine='febus').to_xarray()
    elif dev == 'sintela':
        das_data = xdas.open_mfdataarray(source_files, dim='time', tolerance=np.timedelta64(30, "ms"), engine='sintela').to_xarray()
    else:
        raise Exception('Uredjaj!!!!')
    

    delta_x = das_data.distance[1].data - das_data.distance[0].data
    delta_t = pd.to_timedelta(das_data.time[1].data - das_data.time[0].data).total_seconds()

    with h5py.File(new_file, 'w') as nfl:
        begin_time_str = str(das_data.time[0].data)

        ds = nfl.create_dataset('data', data=das_data.data.T, dtype=das_data.data.dtype)

        ds.attrs['dt_s'] = delta_t
        ds.attrs['dx_m'] = delta_x
        ds.attrs['begin_time'] = begin_time_str

    




def _new_file(source_files, dev, new_filename=None):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(f'{SCRIPT_DIR}/das_data_tmp', exist_ok=True)

    converted_dir = SCRIPT_DIR + '/das_data_tmp/'
    if len(source_files) > 1:
        base, ext = os.path.splitext(os.path.basename(source_files[0]))
        base += '_extended'
    else:
        base, ext = os.path.splitext(os.path.basename(source_files[0]))
    if new_filename == None:
        new_filename = f"{converted_dir}{base.split('/')[-1]}{ext}"
    else:
        new_filename = f'{converted_dir}{new_filename}{ext}'

    if os.path.exists(new_filename):
        os.remove(new_filename)
    _convert_file(source_files=source_files, new_file=new_filename, dev=dev)

    return new_filename








def pick_arrivals(files, dev, pathToEqNet, workers=0):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    converted_file = _new_file(source_files=files, dev=dev)
    
    ngpu = torch.cuda.device_count()
    base_cmd = f"{pathToEqNet} --model phasenet_das --data_list={SCRIPT_DIR}/files.txt  --data_path {SCRIPT_DIR}/das_data_converted --result_path {SCRIPT_DIR}/results --format=h5  --batch_size 1 --workers {workers}"
    
    picks_file = os.path.splitext(os.path.basename(converted_file))[0]
    
    picks_file_path = f'{SCRIPT_DIR}/results/picks_phasenet_das/{picks_file}.csv'
    if os.path.exists(picks_file_path):
        os.remove(picks_file_path)
    
    with open(f"{SCRIPT_DIR}/files.txt", "w") as f:
        f.write(converted_file)
    
    if ngpu == 0:
        cmd = f"python {base_cmd} --device cpu"
    elif ngpu == 1:
        cmd = f"python {base_cmd}"
    else:
        cmd = f"torchrun --nproc_per_node {ngpu} {base_cmd}"
    
    print(cmd)
    os.system(cmd)
    
    picks = pd.read_csv(f"{SCRIPT_DIR}/results/picks_phasenet_das/{picks_file}.csv")
    
    if os.path.exists(converted_file):
        os.remove(converted_file)