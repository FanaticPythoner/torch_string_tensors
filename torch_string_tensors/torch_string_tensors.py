import warnings
import torch
import torch.nn.functional as F
from typing import List, Tuple

import numpy as np

###############################################################################
# WARNING SUPPRESSION: "The given buffer is not writable..."
###############################################################################
# We intentionally skip the warning about "non-writable buffer" because
# we are okay with PyTorch writing to that buffer if needed.
# In practice, we're reading from it for encoding, which is safe.

warnings.filterwarnings(
    "ignore",
    message="The given buffer is not writable, and PyTorch does not support non-writable tensors",
)


###############################################################################
# Low-level utility: read a CPU-contiguous tensor's bytes via ctypes
###############################################################################

def _tensor_slice_to_bytes(row_slice: torch.Tensor) -> bytes:
    """
    Helper that:
      1) Moves row_slice to CPU if needed.
      2) Upcasts bfloat16 -> float32 if row_slice.dtype == torch.bfloat16.
      3) Converts to .numpy().tobytes() for maximum speed on typical small slices.
    """
    if row_slice.device.type != 'cpu':
        row_slice = row_slice.to('cpu', copy=False)

    if row_slice.dtype == torch.bfloat16:
        row_slice = row_slice.float()
    if row_slice.dtype != torch.uint8:
        row_slice = row_slice.to(torch.uint8)

    return row_slice.numpy().tobytes()


###############################################################################
# Functional API
###############################################################################

def list_to_tensor(
    strings: List[str],
    encoding: str = "utf-8",
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of strings into a 2D padded codes tensor plus a
    1D lengths tensor, using purely vectorized indexing for the (n x m) fill.
    """
    n = len(strings)
    if n == 0:
        return (torch.zeros(0, 0, dtype=torch.uint8, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    offsets = torch.empty(n + 1, dtype=torch.long, device=device)
    offsets[0] = 0
    big_data = bytearray()
    offset_val = 0
    for i, s in enumerate(strings):
        b = s.encode(encoding)
        big_data.extend(b)
        offset_val += len(b)
        offsets[i + 1] = offset_val

    lengths = offsets[1:] - offsets[:-1]
    m = lengths.max().item()
    total_bytes = offsets[-1].item()

    row_ids = torch.arange(n, dtype=torch.long, device=device).unsqueeze(1).expand(n, m)
    col_ids = torch.arange(m, dtype=torch.long, device=device).unsqueeze(0).expand(n, m)
    index_2d = offsets[:-1][row_ids] + col_ids
    index_2d.clamp_(0, total_bytes - 1)

    # Use NumPy to interpret big_data as uint8, then directly create a GPU/CPU tensor.
    base_tensor = torch.tensor(
        np.frombuffer(big_data, dtype=np.uint8),
        dtype=torch.uint8,
        device=device
    )
    # Now index directly on this base_tensor which is already on device
    codes_2D = base_tensor[index_2d]

    mask = col_ids >= lengths[row_ids]
    codes_2D[mask] = 0

    return codes_2D, lengths


def tensor_to_list(
    codes_2D: torch.Tensor,
    lengths: torch.Tensor,
    encoding: str = "utf-8"
) -> List[str]:
    """
    Decode a 2D padded codes tensor plus lengths back into a list of strings,
    using the ctypes-based extraction of raw bytes.
    """
    n = codes_2D.shape[0]
    if n == 0:
        return []

    results = []
    for i in range(n):
        length_i = lengths[i].item()
        row_slice = codes_2D[i, :length_i]
        row_bytes = _tensor_slice_to_bytes(row_slice)
        s = row_bytes.decode(encoding)
        results.append(s)

    return results


def string_to_tensor(
    s: str,
    encoding: str = "utf-8",
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    """
    Convert a single string to a 1D PyTorch tensor of dtype=torch.uint8,
    with minimal overhead and direct device construction if possible.
    """
    b = s.encode(encoding)
    if len(b) == 0:
        return torch.empty((0,), dtype=torch.uint8, device=device)

    # Convert to NumPy on CPU, then create the final tensor directly on device
    base_tensor = torch.tensor(
        np.frombuffer(b, dtype=np.uint8),
        dtype=torch.uint8,
        device=device
    )
    return base_tensor


def tensor_to_string(
    codes_1D: torch.Tensor,
    encoding: str = "utf-8"
) -> str:
    """
    Decode a single 1D PyTorch tensor of dtype=torch.uint8 into a Python string,
    via _tensor_slice_to_bytes with ctypes.
    """
    if codes_1D.numel() == 0:
        return ""
    data_bytes = _tensor_slice_to_bytes(codes_1D)
    return data_bytes.decode(encoding)


def string_to_tensor_asview(
    s: str,
    encoding: str = "utf-8",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    Alternate implementation using torch.as_tensor(memoryview(...)).
    """
    b = s.encode(encoding)
    if len(b) == 0:
        return torch.empty((0,), dtype=torch.uint8, device=device)

    # "memoryview" on the bytes object
    mem_view = memoryview(b)

    # We can request dtype=torch.uint8 here.
    # as_tensor will (hopefully) avoid a copy if it can.
    t = torch.as_tensor(mem_view, dtype=torch.uint8)

    # If we are on CPU device by default, no copy is made (in principle).
    # If we must move it to GPU, that will cause a copy.
    if device != "cpu":
        t = t.to(device, non_blocking=True)

    return t


def patch_functional() -> None:
    """
    Monkey-patch torch.nn.functional with:
      list_to_tensor, tensor_to_list,
      string_to_tensor, tensor_to_string
    """
    F.list_to_tensor = list_to_tensor
    F.tensor_to_list = tensor_to_list
    F.string_to_tensor = string_to_tensor
    F.tensor_to_string = tensor_to_string
