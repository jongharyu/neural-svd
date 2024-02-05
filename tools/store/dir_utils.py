import os


def get_latest_run_id(dir_name):
    # assume dir names are run_0, run_1, run_2, etc. 
    curr_run = 0
    for subdir in os.listdir(dir_name):
        full_path = os.path.join(dir_name, subdir)
        if os.path.isdir(full_path):
            run_num = int(subdir.split('_')[-1])
            if run_num > curr_run:
                curr_run = run_num
    return curr_run + 1
