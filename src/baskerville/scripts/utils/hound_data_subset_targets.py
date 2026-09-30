import argparse
import os
import pandas as pd
import numpy as np

try:
    import slurmrunner
except ModuleNotFoundError:
    slurmrunner = None


def subset_targets_dataset(og_data_dir, new_data_dir, old_indices_file, use_slurm=True):
    """
    Create a new dataset directory containing only a subset of user-specified targets.

    Args:
      og_data_dir (str): Original dataset directory.
      new_data_dir (str): Output dataset directory.
      old_indices_file (str): Path to .txt file where the lines contain indices of targets to keep.
      use_slurm (bool): Whether to use SLURM for zarr processing. If False, runs locally.

    Returns:
      None
    """

    # --- MAKE NEW DIR ---

    os.makedirs(new_data_dir, exist_ok=True)

    # --- COPY targets.txt AND SUBSET TARGETS OF INTEREST ---

    # ** load old indices **
    old_indices = np.loadtxt(old_indices_file).astype(int)
    # Ensure old_indices is always a 1D array (loadtxt returns scalar for single value)
    old_indices = np.atleast_1d(old_indices)

    # ** load original targets and subset to targets of interest **
    targets_df = pd.read_table(f"{og_data_dir}/targets.txt", sep="\t", index_col=0)
    sub_targets_df = targets_df.loc[old_indices].copy()

    # ** get  map from old to new indices **
    new_indices = np.arange(0, len(old_indices))
    old_to_new_map = dict(zip(old_indices, new_indices))

    # ** reindex **
    sub_targets_df["index"] = new_indices
    sub_targets_df = sub_targets_df.set_index("index")
    sub_targets_df["strand_pair"] = [
        old_to_new_map[x] for x in np.array(sub_targets_df["strand_pair"])
    ]

    # ** write out **
    fileout = f"{new_data_dir}/targets.txt"
    sub_targets_df.to_csv(fileout, sep="\t")

    # --- COPY sequences.bed ---
    command = f"cp {og_data_dir}/sequences.bed {new_data_dir}/sequences.bed"
    os.system(command)

    # --- COPY OVER statistics.json ---

    # ** copy **
    command = f"cp {og_data_dir}/statistics.json {new_data_dir}/statistics.json"
    os.system(command)

    # ** read in **
    import json

    with open(f"{new_data_dir}/statistics.json") as stats_open:
        stats = json.load(stats_open)

    # ** update num targets **
    stats["num_targets"] = len(sub_targets_df)

    # ** write out **
    with open(f"{new_data_dir}/statistics.json", "w") as f:
        json.dump(stats, f, indent=4)

    # --- COPY examples/ AND SUBSET TARGETS OF INTEREST ---

    num_folds = len([x for x in stats.keys() if "fold" in x])
    if use_slurm and slurmrunner is None:
        print("slurmrunner not installed, running locally")
        use_slurm = False
    if use_slurm:
        # ** get slurm jobs **
        jobs = []
        for fi in range(0, num_folds):
            # ** get command **
            command = (
                f"python -m baskerville.scripts.utils.hound_data_subset_zarr_targets "
                f"--fold_index {fi} "
                f"--old_data_dir {og_data_dir} "
                f"--new_data_dir {new_data_dir} "
                f"--num_workers 8 "
                f"--old_indices {old_indices_file} "
                f"--batch_size 8"
            )

            # ** get slurm job **

            # set up slurm job
            slurm_dir = f"{new_data_dir}/slurm/subset_zarr"
            os.makedirs(slurm_dir, exist_ok=True)
            job_name = f"subset_fold{fi}"
            err_file = f"{slurm_dir}/{job_name}.err"
            out_file = f"{slurm_dir}/{job_name}.out"
            sb_file = f"{slurm_dir}/{job_name}.sb"
            job = slurmrunner.Job(
                command,
                name=job_name,
                out_file=out_file,
                err_file=err_file,
                sb_file=sb_file,
                queue="standard",
                cpu=16,
                mem=32000,
                time="12:0:0",
            )
            jobs.append(job)

        slurmrunner.multi_run(
            jobs,
            max_proc=30,
            verbose=True,
            launch_sleep=1,
            update_sleep=2,
        )
    else:
        # Run locally without SLURM for testing
        from . import hound_data_subset_zarr_targets

        for fi in range(num_folds):
            hound_data_subset_zarr_targets.subset_zarr_for_fold(
                fold_index=fi,
                old_data_dir=og_data_dir,
                new_data_dir=new_data_dir,
                old_indices_file=old_indices_file,
                num_workers=4,
                batch_size=8,
            )


def main():
    """
    Main function for command-line usage.

    Description:
        Given already existing dataset directory produced by hound_data, create a new dataset directory
        containing only a subset of user-specified targets.

    Args:
      data_dir_in (str): Original dataset directory.
      data_dir_out (str): Output dataset directory.
      in_target_indices (str): Path to .txt file where the lines contain indices of targets to keep.

    Example usage:
        python -m baskerville.scripts.utils.hound_data_subset_targets \
            --data_dir_in /path/to/old/data/dir \
            --data_dir_out /path/to/new/data/dir \
            --in_target_indices /path/to/old/indices.txt

        python -m baskerville.scripts.utils.hound_data_subset_targets \
            --data_dir_in data/hg38 \
            --data_dir_out data/hg38_gtex \
            --in_target_indices src/tests/data/hg38_gtex_target_indices.txt

    ToDo:
        - Sequence symlinks: Symlink sequences files instead of copying to save on disk space / run time.
        - Optimize batching / num workers for improved speed.
    """

    # --- PARSE ARGS ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_in", type=str, required=True)
    parser.add_argument("--data_dir_out", type=str, required=True)
    parser.add_argument("--in_target_indices", type=str, required=True)
    parser.add_argument(
        "--local", action="store_true", help="Run locally without SLURM"
    )
    args = parser.parse_args()

    # Call the subset function
    subset_targets_dataset(
        og_data_dir=args.data_dir_in,
        new_data_dir=args.data_dir_out,
        old_indices_file=args.in_target_indices,
        use_slurm=not args.local,
    )


if __name__ == "__main__":
    main()
