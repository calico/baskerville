# Copyright 2023 Calico Life Sciences LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import h5py
import os
import numpy as np
import shutil
from typing import Dict, List, Tuple, Any

from baskerville.snps import write_quantiles


def _get_job_file_path(out_dir: str, job_idx: int, h5f_name: str) -> str:
    """Get the file path for a specific job's HDF5 file."""
    return f"{out_dir}/job{job_idx}/{h5f_name}"


def _get_final_file_path(out_dir: str, h5f_name: str) -> str:
    """Get the file path for the final combined HDF5 file."""
    return f"{out_dir}/{h5f_name}"


def _collect_snp_metadata_across_jobs(
    out_dir: str, num_jobs: int, h5f_name: str
) -> Dict[str, List[Any]]:
    """Collect SNP metadata (chr, pos, ref_allele, alt_allele) across all jobs."""
    all_metadata = {"chr": [], "pos": [], "ref_allele": [], "alt_allele": []}

    for pi in range(num_jobs):
        job_h5_file = _get_job_file_path(out_dir, pi, h5f_name)
        with h5py.File(job_h5_file, "r") as job_h5:
            all_metadata["chr"].extend([c.decode() for c in job_h5["chr"][:]])
            all_metadata["pos"].extend(job_h5["pos"][:])
            all_metadata["ref_allele"].extend(
                [r.decode() for r in job_h5["ref_allele"][:]]
            )
            all_metadata["alt_allele"].extend(
                [a.decode() for a in job_h5["alt_allele"][:]]
            )

    return all_metadata


def _create_snp_metadata_datasets(
    final_h5: h5py.File, metadata: Dict[str, List[Any]]
) -> None:
    """Create SNP metadata datasets in the final HDF5 file."""
    final_h5.create_dataset("chr", data=np.array(metadata["chr"], dtype="S"))
    final_h5.create_dataset("pos", data=np.array(metadata["pos"], dtype="uint32"))
    final_h5.create_dataset(
        "ref_allele", data=np.array(metadata["ref_allele"], dtype="S")
    )
    final_h5.create_dataset(
        "alt_allele", data=np.array(metadata["alt_allele"], dtype="S")
    )


def collect_scores(
    out_dir: str,
    num_jobs: int,
    quantile_copy: bool = False,
    h5f_name: str = "scores.h5",
):
    """Collect parallel SAD jobs' output into one HDF5.

    Args:
        out_dir (str): Output directory.
        num_jobs (int): Number of jobs to combine results from.
        quantile_copy (bool): If True, copy quantiles from job0 instead of recomputing.
    """
    # copy targets files from job0 to final output
    for targets_name in ["targets_cov.txt", "targets_covgene.txt", "targets_gene.txt"]:
        job0_targets_file = f"{out_dir}/job0/{targets_name}"
        if os.path.exists(job0_targets_file):
            shutil.copyfile(job0_targets_file, f"{out_dir}/{targets_name}")

    # Check if this is gene-specific mode by looking at job0
    job0_h5_file = f"{out_dir}/job0/{h5f_name}"
    with h5py.File(job0_h5_file, "r") as job0_h5_open:
        gene_mode = "gene_ids" in job0_h5_open

    if gene_mode:
        _collect_scores_gene(out_dir, num_jobs, quantile_copy, h5f_name)
    else:
        _collect_scores_genome(out_dir, num_jobs, quantile_copy, h5f_name)


def _collect_scores_genome(
    out_dir: str,
    num_jobs: int,
    quantile_copy: bool = False,
    h5f_name: str = "scores.h5",
):
    """Collect parallel SAD jobs' output for gene-agnostic genome-wide scoring mode."""
    assert num_jobs > 0

    # 1. Count total SNPs and get per-job SNP counts
    total_snps, job_snp_counts = _count_items_across_jobs(
        out_dir, num_jobs, h5f_name, "snp"
    )

    # 2. Validate that we have SNPs to process
    if total_snps == 0:
        raise ValueError(
            f"No SNPs found across {num_jobs} jobs in {out_dir}. "
            "This indicates a problem with the input data or job processing."
        )

    # 3. Get final HDF5 file path and open it
    final_h5_file = _get_final_file_path(out_dir, h5f_name)
    with h5py.File(final_h5_file, "w") as final_h5:
        job0_h5_file = _get_job_file_path(out_dir, 0, h5f_name)
        with h5py.File(job0_h5_file, "r") as job0_h5:
            # 5. Collect and create SNP character/string metadata datasets (chr, pos, ref_allele, alt_allele)
            snp_char_metadata = _collect_snp_metadata_across_jobs(
                out_dir, num_jobs, h5f_name
            )
            _create_snp_metadata_datasets(
                final_h5, snp_char_metadata
            )  # Creates chr, pos, ref_allele, alt_allele

            # 6. Collect and create 'snp' dataset (SNP IDs)
            snp_ids_data = _collect_string_datasets_across_jobs(
                out_dir, num_jobs, h5f_name, ["snp"]
            )
            if "snp" in snp_ids_data and snp_ids_data["snp"]:
                final_h5.create_dataset(
                    "snp", data=np.array(snp_ids_data["snp"], dtype="S")
                )
            else:
                # If SNPs were counted but no IDs collected (e.g. all jobs had empty 'snp' datasets),
                # create an empty 'snp' dataset of the correct total size.
                final_h5.create_dataset("snp", shape=(total_snps,), dtype="S")

            # 7. Identify all keys from job0 that need to be initialized in the final HDF5.
            score_keys = _get_score_dataset_names(job0_h5)
            _initialize_score_datasets(final_h5, job0_h5, score_keys, total_snps)

        # 9. Populate the concatenated score datasets
        current_snp_idx = 0
        for pi in range(num_jobs):
            job_h5_file = _get_job_file_path(out_dir, pi, h5f_name)
            with h5py.File(job_h5_file, "r") as job_h5:
                num_snps_in_job = job_snp_counts[pi]

                for key in score_keys:
                    if key in job_h5 and key in final_h5:
                        if job_h5[key].shape[0] == num_snps_in_job:
                            final_h5[key][
                                current_snp_idx : current_snp_idx + num_snps_in_job
                            ] = job_h5[key][:]
                        else:
                            raise ValueError(
                                f"{job_h5_file} dataset '{key}' has mismatched shape. "
                                f"Expected first dimension {num_snps_in_job}, got {job_h5[key].shape[0]}. "
                                "Remove this file and rerun."
                            )
                current_snp_idx += num_snps_in_job

        # 10. Recompute quantile statistics from complete dataset
        norm_file = job0_h5_file if quantile_copy else None
        write_quantiles(final_h5, score_keys, norm_file)


def _collect_scores_gene(
    out_dir: str,
    num_jobs: int,
    quantile_copy: bool = False,
    h5f_name: str = "scores.h5",
):
    """Collect parallel SAD jobs\' output for gene-specific SNP scoring mode."""
    assert num_jobs > 0

    # 1. Collect all SNP IDs and count SNPs per job
    snp_ids_data = _collect_string_datasets_across_jobs(
        out_dir, num_jobs, h5f_name, ["snp"]
    )
    all_snp_ids = snp_ids_data.get("snp", [])
    total_snps, job_snp_counts = _count_items_across_jobs(
        out_dir, num_jobs, h5f_name, "snp"
    )

    # 2. Validate that we have SNPs to process
    if len(all_snp_ids) == 0:
        raise ValueError(
            f"No SNPs found across {num_jobs} jobs in {out_dir}. "
            "This indicates a problem with the input data or job processing."
        )

    # 3. Collect all unique gene IDs
    gene_ids_data = _collect_string_datasets_across_jobs(
        out_dir, num_jobs, h5f_name, ["gene_ids"]
    )
    all_gene_ids_set = set(gene_ids_data.get("gene_ids", []))
    all_gene_ids = sorted(list(all_gene_ids_set))

    # Create global SNP/gene to index mappings
    snp_to_global_idx = {snp_id: idx for idx, snp_id in enumerate(all_snp_ids)}
    gene_to_global_idx = {gene_id: idx for idx, gene_id in enumerate(all_gene_ids)}
    assert len(all_gene_ids) > 0, (
        "No gene IDs found across jobs. Ensure jobs have valid gene data."
    )
    assert len(all_snp_ids) > 0, (
        "No SNP IDs found across jobs. Ensure jobs have valid SNP data."
    )

    # 3. Count total SNP-gene pairs and get per-job pair counts
    total_snp_gene_pairs, job_pair_counts = _count_items_across_jobs(
        out_dir, num_jobs, h5f_name, "snp_idx"
    )
    assert total_snp_gene_pairs > 0, (
        "No SNP-gene pairs found across jobs. Ensure jobs have valid SNP-gene data."
    )

    # 4. Initialize final HDF5 file
    final_h5_file = _get_final_file_path(out_dir, h5f_name)
    with h5py.File(final_h5_file, "w") as final_h5:
        job0_h5_file = _get_job_file_path(out_dir, 0, h5f_name)
        with h5py.File(job0_h5_file, "r") as job0_h5:
            # 5. Create SNP metadata datasets (snp, chr, pos, ref_allele, alt_allele)
            final_h5.create_dataset("snp", data=np.array(all_snp_ids, dtype="S"))
            snp_char_metadata = _collect_snp_metadata_across_jobs(
                out_dir, num_jobs, h5f_name
            )
            _create_snp_metadata_datasets(final_h5, snp_char_metadata)

            # 7. Create unified gene_ids dataset
            final_h5.create_dataset("gene_ids", data=np.array(all_gene_ids, dtype="S"))

            # 8. Discover and partition score datasets by indexing scheme
            exclude_for_scores = [
                "snp",
                "chr",
                "pos",
                "ref_allele",
                "alt_allele",
                "targets",
                "quantiles",
                "gene_ids",
                "snp_idx",
                "gene_idx",
            ]
            all_score_keys = _get_score_dataset_names(
                job0_h5, exclude_keys=exclude_for_scores
            )
            # cov/ keys are SNP-indexed; covgene/ and gene/ keys are pair-indexed
            snp_score_keys = [k for k in all_score_keys if k.startswith("cov/")]
            pair_score_keys = [
                k
                for k in all_score_keys
                if k.startswith("covgene/") or k.startswith("gene/")
            ]

            # Initialize datasets with appropriate dimensions
            _initialize_score_datasets(final_h5, job0_h5, snp_score_keys, total_snps)
            _initialize_score_datasets(
                final_h5, job0_h5, pair_score_keys, total_snp_gene_pairs
            )

        # 9. Create and populate remapped snp_idx and gene_idx arrays, and copy scores
        final_snp_idx = np.zeros(total_snp_gene_pairs, dtype=np.int32)
        final_gene_idx = np.zeros(total_snp_gene_pairs, dtype=np.int32)

        current_pair_offset = 0
        current_snp_offset = 0

        for pi in range(num_jobs):
            job_h5_file = _get_job_file_path(out_dir, pi, h5f_name)
            with h5py.File(job_h5_file, "r") as job_h5:
                num_pairs_in_job = job_pair_counts[pi]
                num_snps_in_job = job_snp_counts[pi]

                # Copy SNP-indexed score data
                for key in snp_score_keys:
                    if key in job_h5 and key in final_h5:
                        final_h5[key][
                            current_snp_offset : current_snp_offset + num_snps_in_job
                        ] = job_h5[key][:]

                if num_pairs_in_job > 0:
                    # Remap SNP-gene pair indices
                    job_local_snp_ids = [s.decode() for s in job_h5["snp"][:]]
                    job_local_gene_ids = [g.decode() for g in job_h5["gene_ids"][:]]

                    job_local_snp_idx_to_global = {
                        local_idx: snp_to_global_idx[snp_id]
                        for local_idx, snp_id in enumerate(job_local_snp_ids)
                    }
                    job_local_gene_idx_to_global = {
                        local_idx: gene_to_global_idx[gene_id]
                        for local_idx, gene_id in enumerate(job_local_gene_ids)
                    }

                    job_h5_snp_indices = job_h5["snp_idx"][:]
                    job_h5_gene_indices = job_h5["gene_idx"][:]

                    for i in range(num_pairs_in_job):
                        final_snp_idx[current_pair_offset + i] = (
                            job_local_snp_idx_to_global[job_h5_snp_indices[i]]
                        )
                        final_gene_idx[current_pair_offset + i] = (
                            job_local_gene_idx_to_global[job_h5_gene_indices[i]]
                        )

                    # Copy pair-indexed score data
                    for key in pair_score_keys:
                        if key in job_h5 and key in final_h5:
                            final_h5[key][
                                current_pair_offset : current_pair_offset
                                + num_pairs_in_job
                            ] = job_h5[key][:]

                    current_pair_offset += num_pairs_in_job

                current_snp_offset += num_snps_in_job

        # Save the remapped index arrays
        final_h5.create_dataset("snp_idx", data=final_snp_idx)
        final_h5.create_dataset("gene_idx", data=final_gene_idx)

        # 10. Recompute quantile statistics from complete dataset
        norm_file = job0_h5_file if quantile_copy else None
        write_quantiles(final_h5, snp_score_keys + pair_score_keys, norm_file)


def check_progress_h5(
    h5_file: str, expected_status: str = "completed", verbose: bool = False
) -> bool:
    """Check if an HDF5 file has the expected progress status.

    Args:
        h5_file (str): HDF5 file path.
        expected_status (str): Expected progress status to check for.
        verbose (bool): If True, print diagnostic messages.

    Returns:
        bool: True if file exists and has expected status, False otherwise.
    """
    if not os.path.isfile(h5_file):
        if verbose:
            print(f"File {h5_file} does not exist")
        return False

    try:
        with h5py.File(h5_file, "r") as h5_open:
            if "progress_status" not in h5_open:
                if verbose:
                    print(f"{h5_file} has no progress_status dataset")
                return False

            status = h5_open["progress_status"][()].decode("utf-8")
            if status != expected_status:
                if verbose:
                    print(
                        f"{h5_file} has status '{status}', expected '{expected_status}'"
                    )
                return False

            return True
    except Exception as e:
        if verbose:
            print(f"Error reading progress from {h5_file}: {e}")
        return False


def _count_items_across_jobs(
    out_dir: str, num_jobs: int, h5f_name: str, dataset_name: str
) -> Tuple[int, List[int]]:
    """Count total items and per-job counts for a dataset across all jobs."""
    total_count = 0
    job_counts = []

    for pi in range(num_jobs):
        job_h5_file = _get_job_file_path(out_dir, pi, h5f_name)
        with h5py.File(job_h5_file, "r") as job_h5:
            count = len(job_h5[dataset_name])
            total_count += count
            job_counts.append(count)

    return total_count, job_counts


def _collect_string_datasets_across_jobs(
    out_dir: str, num_jobs: int, h5f_name: str, dataset_names: List[str]
) -> Dict[str, List[str]]:
    """Collect string datasets across all jobs."""
    collected_data = {name: [] for name in dataset_names}

    for pi in range(num_jobs):
        job_h5_file = _get_job_file_path(out_dir, pi, h5f_name)
        with h5py.File(job_h5_file, "r") as job_h5:
            for name in dataset_names:
                if name in job_h5:
                    decoded_data = [item.decode() for item in job_h5[name][:]]
                    collected_data[name].extend(decoded_data)

    return collected_data


def _get_score_dataset_names(
    job0_h5: h5py.File, exclude_keys: List[str] = None
) -> List[str]:
    """Get the names of score datasets from job0, excluding specified keys.

    Walks the full HDF5 tree to find datasets (including nested ones like gene/logSUM).
    Excludes _quantiles keys since they will be created by write_quantiles().
    """
    if exclude_keys is None:
        exclude_keys = []

    base_exclude = [
        "snp",
        "chr",
        "pos",
        "ref_allele",
        "alt_allele",
        "targets",
        "quantiles",
        "progress_status",
    ]
    all_exclude = set(base_exclude + exclude_keys)

    score_keys = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            if name not in all_exclude and not name.endswith("_quantiles"):
                score_keys.append(name)

    job0_h5.visititems(_visit)
    return score_keys


def _initialize_score_datasets(
    final_h5: h5py.File, job0_h5: h5py.File, score_keys: List[str], total_items: int
) -> None:
    """Initialize score datasets in the final HDF5 file."""
    for key in score_keys:
        if key not in job0_h5:
            continue
        dataset = job0_h5[key]
        if dataset.ndim == 0:
            continue
        final_h5.create_dataset(
            key,
            shape=(total_items,) + dataset.shape[1:],
            dtype=dataset.dtype,
        )
