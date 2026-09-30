import torch


def check_mixed_precision():
    """Check if the current GPU supports native mixed precision training."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    device = torch.cuda.current_device()
    capabilities = torch.cuda.get_device_capability(device)

    # Volta (SM70), Turing (SM75), Ampere (SM80+) support native mixed precision
    return capabilities >= (7, 0)
