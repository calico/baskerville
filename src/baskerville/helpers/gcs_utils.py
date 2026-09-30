# Adapted from calico/baskerville-tf GCS helpers.

import json
import os
import re
import tempfile
from os.path import exists, join, isfile
from re import match

from google.cloud.storage import Client


def download_from_gcs(gcs_path: str, local_path: str, bytes=True) -> None:
    """
    Downloads a file from GCS
    Args:
        gcs_path: string path to GCS file to download
        local_path: string path to download to
        bytes: boolean flag indicating if gcs file contains bytes

    Returns: None

    """
    storage_client = Client()
    write_mode = "wb" if bytes else "w"
    with open(local_path, write_mode) as o:
        storage_client.download_blob_to_file(gcs_path, o)


def read_json_gcs(gcs_uri: str) -> dict:
    """Download a JSON object from GCS and parse it into a dict.

    Used to read small control files (statistics.json, progress.json, ...)
    that live alongside a run's data/outputs in GCS.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        download_from_gcs(gcs_uri, tmp_path, bytes=True)
        with open(tmp_path) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def download_folder_from_gcs(
    gcs_dir: str, local_dir: str, bytes=True, exclude_regex: str | None = None
) -> None:
    """
    Downloads a whole folder from GCS
    Args:
        gcs_dir: string path to GCS folder to download
        local_dir: string path to download to
        bytes: boolean flag indicating if gcs file contains bytes
        exclude_regex: if given, blobs whose path relative to gcs_dir matches
            this regex (``re.search``) are skipped — e.g. ``r".*\\.pth$"`` to
            skip checkpoint weights.

    Returns: None

    """
    storage_client = Client()
    write_mode = "wb" if bytes else "w"
    if not is_gcs_path(gcs_dir):
        raise ValueError(f"gcs_dir is not a valid GCS path: {gcs_dir}")
    bucket_name, gcs_object_prefix = split_gcs_uri(gcs_dir)
    # Get the bucket from the client.
    bucket = storage_client.bucket(bucket_name)

    exclude = re.compile(exclude_regex) if exclude_regex else None

    # Ensure local folder exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # Match only objects strictly beneath this directory. A bare prefix would
    # also catch sibling objects that share the name as a string prefix (e.g.
    # listing ".../f0c0/train" would also return ".../f0c0/train.err"), which
    # then yields a "../" relpath and a bogus download. The trailing slash
    # pins the match to a path boundary.
    dir_prefix = gcs_object_prefix.rstrip("/") + "/"
    # List all blobs with the given prefix (i.e., folder path).
    blobs = bucket.list_blobs(prefix=dir_prefix)
    # Download each blob.
    for blob in blobs:
        # Skip the directory placeholder object, if any.
        if blob.name == dir_prefix:
            continue
        # Compute the full path to which we'll download the blob.
        blob_rel_path = os.path.relpath(blob.name, dir_prefix)
        # Skip excluded paths (e.g. .pth weights for the light periodic mirror).
        if exclude is not None and exclude.search(blob_rel_path):
            continue
        local_blob_path = os.path.join(local_dir, blob_rel_path)

        # Ensure the local directory structure exists
        local_blob_dir = os.path.dirname(local_blob_path)
        if not os.path.exists(local_blob_dir):
            os.makedirs(local_blob_dir)
        download_from_gcs(join(gcs_dir, blob_rel_path), local_blob_path, bytes=bytes)


def sync_dir_to_gcs(
    local_dir: str, gcs_dir: str, verbose=False, recursive=False
) -> None:
    """
    Copies all files in a local directory to the gcs directory
    Args:
        local_dir: string local directory path to upload from
        gcs_dir: string GCS destination path. Will create folders that do not exist.
        verbose: boolean flag to print logging statements
        recursive: boolean flag to recursively upload files in subdirectories

    Returns: None

    """
    storage_client = Client()
    if not is_gcs_path(gcs_dir):
        raise ValueError(f"gcs_dir is not a valid GCS path: {gcs_dir}")

    if not exists(local_dir):
        raise FileNotFoundError(f"local_dir does not exist: {local_dir}")

    local_files = os.listdir(local_dir)
    bucket_name, gcs_object_prefix = split_gcs_uri(gcs_dir)
    bucket = storage_client.bucket(bucket_name)

    for filename in local_files:
        gcs_object_name = join(gcs_object_prefix, filename)
        local_file = join(local_dir, filename)
        if recursive and not isfile(local_file):
            sync_dir_to_gcs(
                local_file,
                f"gs://{join(bucket_name, gcs_object_name)}",
                verbose=verbose,
                recursive=recursive,
            )
        elif not isfile(local_file):
            pass
        else:
            blob = bucket.blob(gcs_object_name)
            if verbose:
                print(
                    f"Uploading {local_file} to gs://{join(bucket_name, gcs_object_name)}"
                )
            blob.upload_from_filename(local_file)


def upload_folder_gcs(local_dir: str, gcs_dir: str) -> None:
    """
    Copies all files in a local directory to the gcs directory
    Args:
        local_dir: string local directory path to upload from
        gcs_dir: string GCS destination path. Will create folders that do not exist.
    Returns: None
    """
    storage_client = Client()
    bucket_name = gcs_dir.split("//")[1].split("/")[0]
    gcs_object_prefix = "/".join(gcs_dir.split("//")[1].split("/")[1:])
    local_prefix = local_dir.split("/")[-1]
    bucket = storage_client.bucket(bucket_name)
    for filename in os.listdir(local_dir):
        gcs_object_name = f"{gcs_object_prefix}/{local_prefix}/{filename}"
        local_file = join(local_dir, filename)
        blob = bucket.blob(gcs_object_name)
        blob.upload_from_filename(local_file)


def split_gcs_uri(gcs_uri: str) -> tuple:
    """
    Splits a GCS bucket and object_name from a GCS URI
    Args:
        gcs_uri: string GCS Uri follows the format gs://$BUCKET_NAME/OBJECT_NAME

    Returns: bucket_name, object_name
    """
    matches = match("gs://(.*?)/(.*)", gcs_uri)
    if matches:
        return matches.groups()
    else:
        raise ValueError(
            f"{gcs_uri} does not match expected format: gs://BUCKET_NAME/OBJECT_NAME"
        )


def delete_from_gcs(gcs_uri: str) -> None:
    """Delete a single object from GCS. No-op if it doesn't exist."""
    storage_client = Client()
    bucket_name, object_name = split_gcs_uri(gcs_uri)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    if blob.exists():
        blob.delete()


def is_gcs_path(gcs_path: str) -> bool:
    """
    Returns True if the string passed starts with gs://
    Args:
        gcs_path: string path to check

    Returns: Boolean flag indicating the gcs_path starts with gs://

    """
    return gcs_path.startswith("gs://")


def download_rename_inputs(filepath: str, temp_dir: str, is_dir: bool = False) -> str:
    """
    Download file from gcs to local dir
    Args:
        filepath: GCS Uri follows the format gs://$BUCKET_NAME/OBJECT_NAME
        temp_dir: local dir to download to
        is_dir: boolean flag indicating if the filepath is a directory
    Returns: new filepath in the local machine
    """
    if is_dir:
        dir_name = filepath.split("/")[-1]
        download_folder_from_gcs(filepath, f"{temp_dir}/{dir_name}")
        return f"{temp_dir}/{dir_name}"
    else:
        _, filename = split_gcs_uri(filepath)
        if "/" in filename:
            filename = filename.split("/")[-1]
        download_from_gcs(filepath, f"{temp_dir}/{filename}")
        return f"{temp_dir}/{filename}"


def gcs_file_exist(gcs_path: str) -> bool:
    """
    check if a file exist in gcs
    params: gcs_path
    returns: true/false
    """
    storage_client = Client()
    bucket, filename = split_gcs_uri(gcs_path)
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    return blob.exists()
