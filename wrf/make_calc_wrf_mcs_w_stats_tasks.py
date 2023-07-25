"""
Make task list and slurm scripts for calculating MCS W statistics.
"""
import sys
import numpy as np
import xarray as xr
import yaml
import textwrap
import subprocess

if __name__ == "__main__":

    run_name = sys.argv[1]
    
    code_dir = '/global/homes/f/feng045/program/iclass/re2/amazon_hvmixing/wrf/'
    code_name = f'calc_wrf_mcs_w_stats_bytracks.py'
    config_file = f'config_w_{run_name}.yml'
    slurm_filename = f'slurm.submit_w_stats_{run_name}.sbat'

    submit_job = False

    # Number of tracks to process per part
    ntracks_part = 3
    # Set the number of digits for 0 padding
    # This should be set to the digit for the maximum number of tracks
    # e.g., ntracks = 32138, digits = 5
    digits = 3

    # Get inputs from configuration file
    stream = open(config_file, 'r')
    config = yaml.full_load(stream)
    track_period = config['track_period']
    mcsmask_dir = config['mcsmask_dir'] + track_period + '/'
    mcsstats_dir = config['mcsstats_dir']
    statsfile_basename = config['statsfile_basename']

    # Track statistics file
    trackstats_file = f"{mcsstats_dir}{statsfile_basename}{track_period}.nc"

    # Read MCS statistics
    mcs_file = f"{mcsstats_dir}{statsfile_basename}{track_period}.nc"
    dsm = xr.open_dataset(mcs_file)
    # Subset MCS times to reduce array size
    # Most valid MCS data are within 0:ntimes_max
    # dsm = dsm.isel(times=slice(0, ntimes_max))
    ntracks_all = dsm.sizes['tracks']

    # Number of parts
    nparts = np.floor(ntracks_all / ntracks_part).astype(int)
    # Make a list for track start/end 
    track_start = []
    track_end = []
    for ii in range(0, nparts): 
        track_start.append(ii*ntracks_part)
        track_end.append((ii+1)*ntracks_part-1)
    # Add the last part to the list
    track_start.append(track_end[-1]+1)
    track_end.append(ntracks_all)
    # Update total number of parts
    nparts = len(track_start)

    # Create the list of job tasks needed by SLURM...
    syear = track_period[0:4]
    task_filename = f'tasks_w_stats_{run_name}.txt'
    task_file = open(task_filename, "w")
    ntasks = 0

    for ipart in range(0, nparts): 
        cmd = f'python {code_name} {config_file} ' \
            f'{track_start[ipart]} {track_end[ipart]} {digits}'
        # print(cmd)
        task_file.write(f"{cmd}\n")
        ntasks += 1
    task_file.close()
    print(task_filename)

    # Create a SLURM submission script for the above task list...
    slurm_file = open(slurm_filename, "w")
    text = f"""\
        #!/bin/bash
        #SBATCH -A m1657
        #SBATCH -J {run_name}
        #SBATCH -t 00:30:00
        #SBATCH -q shared
        #SBATCH -C cpu
        ##SBATCH --nodes=1
        #SBATCH --ntasks-per-node=32
        ##SBATCH --exclusive
        #SBATCH --output=log_w_stats_{run_name}_%A_%a.log
        #SBATCH --mail-type=END
        #SBATCH --mail-user=zhe.feng@pnnl.gov
        #SBATCH --array=1-{ntasks}

        date
        source activate /global/common/software/m1867/python/py310

        cd {code_dir}

        # Takes a specified line ($SLURM_ARRAY_TASK_ID) from the task file
        LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p {task_filename})
        echo $LINE
        # Run the line as a command
        $LINE

        date
        """
    slurm_file.writelines(textwrap.dedent(text))
    slurm_file.close()
    print(slurm_filename)

    # Submit job
    if submit_job == True:
        # cmd = f'sbatch --array=1-{ntasks}%{njobs_run} {slurm_filename}'
        cmd = f'sbatch --array=1-{ntasks} {slurm_filename}'
        print(cmd)
        subprocess.run(f'{cmd}', shell=True)
