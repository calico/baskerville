import torch

############################################################
# mlm
#
# Shared helpers for masked-language-model (MLM) evaluation
# and visualization, so hound_eval_mlm.py and hound_viz_mlm.py
# can't drift apart.
############################################################


def patch_params_for_old_unet(params_model, state_dict):
    """Patch trunk params in place for legacy UnetBorzoiBlock checkpoints.

    Old UnetBorzoiBlock weights are identifiable by ``.conv_depth.`` keys. When
    present, UnetTower / UnetV2Tower trunk blocks must be built with
    ``type="borzoi"`` so the architecture matches the saved weights.

    Args:
        params_model (dict): Model params; its ``trunk`` list is mutated in place.
        state_dict (dict): Loaded checkpoint state dict to inspect.

    Returns:
        bool: True if the checkpoint was detected as old-style and patched.
    """
    is_old_unet = any(".conv_depth." in k for k in state_dict)
    if is_old_unet:
        print(
            "Detected old UnetBorzoiBlock checkpoint — building model with type='borzoi'"
        )
        for block in params_model.get("trunk", []):
            if block.get("name") in ("UnetTower", "UnetV2Tower"):
                block["type"] = "borzoi"
    return is_old_unet


def predict_masked_sequence(
    model, x, seq_length, mask_size, device, mix_dtype, di=0, rc=False
):
    """Iteratively mask and predict every position of a single sequence.

    Masks ``mask_size`` positions per round (DNA channels zeroed) until every
    position has been predicted, keeping the first prediction for each. The
    masking order is padded to a whole number of rounds so all positions are
    covered. With ``rc=True``, averages the forward and reverse-complement
    predictions.

    Args:
        model: SeqNNMod-style module; ``model(x, hi, di).coverage`` returns the
            per-position nucleotide predictions.
        x (Tensor): One-hot input, shape (1, seq_depth, seq_length).
        seq_length (int): Number of positions.
        mask_size (int): Positions masked per round.
        device (str): Torch device for index tensors / autocast.
        mix_dtype: Autocast dtype.
        di (int): Species normalization index.
        rc (bool): Average forward and reverse-complement predictions.

    Returns:
        Tensor: Predicted probabilities, shape (1, 4, seq_length), float32.
    """
    # random masking order, padded so every position is covered in whole rounds
    inds = torch.randperm(seq_length, device=device)
    if seq_length % mask_size > 0:
        missing_n = mask_size - seq_length % mask_size
        missing_inds = torch.randperm(seq_length, device=device)[:missing_n]
        inds = torch.cat([inds, missing_inds], dim=0)

    x_pred = torch.zeros_like(x, dtype=torch.float32)
    b_pred = torch.zeros(seq_length, dtype=torch.bool, device=device)

    while inds.shape[0] > 0:
        ind = inds[:mask_size]
        inds = inds[mask_size:]

        # zero the DNA channels at the masked positions (4 channels only)
        x_masked = x.clone()
        x_masked[0, :4, ind] = 0.0

        # predict forward strand
        with torch.autocast(device_type=device, dtype=mix_dtype):
            yp = model(x_masked, hi=0, di=di).coverage

        # optionally average with the RC prediction. flip(dims=[1, 2]) reverses
        # position and complements channels back to forward coordinates.
        if rc:
            x_masked_rc = torch.flip(x_masked, dims=[1, 2])
            with torch.autocast(device_type=device, dtype=mix_dtype):
                yp_rc = model(x_masked_rc, hi=0, di=di).coverage
            yp = (yp + torch.flip(yp_rc, dims=[1, 2])) / 2.0

        # keep the first prediction for each position
        for j in ind.tolist():
            if not b_pred[j]:
                x_pred[0, :, j] = yp[0, :, j]
                b_pred[j] = True

    return x_pred
