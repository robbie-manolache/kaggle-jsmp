import os
import subprocess
from numpy import setdiff1d

def kernel_output_download(user, kernel_name, local_path = None):
    """
    """

    # set kernel path and record initial files
    kernel_path = "/".join([user, kernel_name])
    files_before = os.listdir(local_path)

    # set up command
    cmd = " ".join(["kaggle kernels output", kernel_path])
    if local_path is not None:
        cmd = cmd + " -p \"%s\""%local_path

    # run command
    subprocess.run(cmd)

    # check change local path contents
    files_after = os.listdir(local_path)
    new_files = list(setdiff1d(files_after, files_before))
    if len(new_files) > 0:
        for f in new_files:
            print("%s has been added to local path"%f)
    else:
        print("No new files added. " +
              "Either existing files were updated " +
              "or kernel does not have any output/exist.")