import os
import sys
import subprocess
import time

############################################################
# utils
#
# Helpful methods that are difficult to categorize.
############################################################


def conda_activate(env):
    """Shell prefix activating conda env `env` in a local/Slurm job, or "" if
    `env` is None (the job inherits the current environment). Sources
    $BASKERVILLE_CONDA (the path to conda.sh) first when set."""
    if not env:
        return ""
    source = os.environ.get("BASKERVILLE_CONDA")
    return (f". {source}; " if source else "") + f"conda activate {env}; "


def exec_par(cmds, max_proc=None, verbose=False):
    """
    Execute the commands in the list 'cmds' in parallel, but
    only running 'max_proc' at a time.

    Args:
        cmds (list): List of commands to execute.
        max_proc (int, optional): Maximum number of processes to run in parallel. Defaults to None.
        verbose (bool, optional): If True, print the commands being executed. Defaults to False.
    """
    total = len(cmds)
    finished = 0
    running = 0
    p = []

    if max_proc == None:
        max_proc = len(cmds)

    if max_proc == 1:
        while finished < total:
            if verbose:
                print(cmds[finished], file=sys.stderr)
            op = subprocess.Popen(cmds[finished], shell=True)
            os.waitpid(op.pid, 0)
            finished += 1

    else:
        while finished + running < total:
            # launch jobs up to max
            while running < max_proc and finished + running < total:
                if verbose:
                    print(cmds[finished + running], file=sys.stderr)
                p.append(subprocess.Popen(cmds[finished + running], shell=True))
                # print('Running %d' % p[running].pid)
                running += 1

            # are any jobs finished
            new_p = []
            for i in range(len(p)):
                # print('POLLING', i, p[i].poll())
                if p[i].poll() != None:
                    running -= 1
                    finished += 1
                else:
                    new_p.append(p[i])

            # if none finished, sleep
            if len(new_p) == len(p):
                time.sleep(1)
            p = new_p

        # wait for all to finish
        for i in range(len(p)):
            p[i].wait()


def detect_model_folds(
    models_dir, cross=0, filename="model_best.pth", train_subdir="train"
):
    """Detect how many sequential folds exist for a given cross index.

    Starting at fold=0, increment while the expected checkpoint file exists.

    Args:
        models_dir (str): Root directory containing model fold/cross subdirectories.
        cross (int, optional): Cross index to inspect. Defaults to 0.
        filename (str, optional): Checkpoint filename to look for. Defaults to "model_best.pth".
        train_subdir (str, optional): Subdirectory containing checkpoint. Defaults to "train".

    Returns:
        int: Number of folds detected (0 if none).
    """
    fold = 0
    while True:
        ckpt_path = os.path.join(models_dir, f"f{fold}c{cross}", train_subdir, filename)
        if os.path.isfile(ckpt_path):
            fold += 1
        else:
            break
    return fold
