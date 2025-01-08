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

def _tensor_slice_to_bytes(row_slice: torch.Tensor) -> bytes:
    """
    Description:
        This function extracts bytes from a given 1D torch.Tensor. The process includes:
        1. Ensuring the tensor is on CPU.
        2. Upcasting from torch.bfloat16 to float if necessary.
        3. Casting to torch.uint8 if needed.
        4. Converting the underlying numpy array to bytes.

    Mathematical details:
        Let k = row_slice.shape[0].
        1) If row_slice is on GPU, it is transferred to CPU.
        2) If row_slice.dtype == bfloat16, we transform row_slice into float32.
        3) If row_slice.dtype != uint8, we cast it to uint8.
        4) We then call row_slice.numpy().tobytes() to retrieve bytes.

    Parameters:
        row_slice (torch.Tensor): A 1D or slice of a torch.Tensor, possibly on GPU, whose bytes are to be read.

    Returns:
        data_bytes (bytes): The bytes representation of the row_slice tensor.
    """

    # --------------------------------------------------------------
    # STEP 1: Check device type.
    # --------------------------------------------------------------
    # If row_slice is not on CPU, move it there without copying if possible.
    if row_slice.device.type != 'cpu':
        row_slice = row_slice.to('cpu', copy=False)

    # ===============
    # Sub step 1.1: Upcast from bfloat16 if needed.
    # ===============
    # If the tensor dtype is torch.bfloat16, convert it to float32.
    if row_slice.dtype == torch.bfloat16:
        row_slice = row_slice.float()

    # ===============
    # Sub step 1.2: Convert to uint8 if not already.
    # ===============
    # Casting to torch.uint8 so we can retrieve the buffer in byte form.
    if row_slice.dtype != torch.uint8:
        row_slice = row_slice.to(torch.uint8)

    # ===============
    # Sub step 1.3: Convert to numpy and get the raw bytes.
    # ===============
    # row_slice.numpy().tobytes() returns the raw buffer as Python bytes.
    return row_slice.numpy().tobytes()


def list_to_tensor(
    strings: List[str],
    encoding: str = "utf-8",
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
        Converts a list of Python strings into two tensors:
        1) codes_2D (n x m): An n-by-m padded matrix of uint8 codes.
        2) lengths (n): A 1D tensor containing the length of each string.

    Mathematical details:
        Let n = len(strings).
        Define offsets as follows (in ASCII):
            offsets[0] = 0
            offsets[i+1] = offsets[i] + len(strings[i])
        for i in {0, 1, ..., n-1}.

        Then lengths[i] = offsets[i+1] - offsets[i].
        Let m = max(lengths). We create codes_2D of shape (n, m).
        Finally:
            codes_2D[i, j] = base_tensor[offsets[i] + j]   if j < lengths[i]
            codes_2D[i, j] = 0                            otherwise

    Parameters:
        strings (List[str]): A list of Python strings to be encoded.
        encoding (str): The text encoding to use for each string. Default is "utf-8".
        device (torch.device): The device on which to place the output tensors.

    Returns:
        codes_2D (torch.Tensor): A padded 2D uint8 tensor containing encoded bytes.
        lengths (torch.Tensor): A 1D tensor containing the lengths of each string.
    """

    # --------------------------------------------------------------
    # STEP 1: Handle empty list case.
    # --------------------------------------------------------------
    # If there are no strings, return empty tensors of correct shape.
    n = len(strings)
    if n == 0:
        return (torch.zeros(0, 0, dtype=torch.uint8, device=device),
                torch.zeros(0, dtype=torch.long, device=device))

    # --------------------------------------------------------------
    # STEP 2: Build offsets and big_data buffer.
    # --------------------------------------------------------------
    # ===============
    # Sub step 2.1: Create offsets tensor.
    # ===============
    offsets = torch.empty(n + 1, dtype=torch.long, device=device)
    offsets[0] = 0

    # ===============
    # Sub step 2.2: Accumulate encoded byte lengths into big_data.
    # ===============
    # big_data is a bytearray that accumulates all encoded bytes sequentially.
    big_data = bytearray()
    offset_val = 0
    for i, s in enumerate(strings):
        b = s.encode(encoding)  # Encode string to bytes using Python's encode.
        big_data.extend(b)      # Extend our bytearray with the encoded bytes.
        offset_val += len(b)    # Keep track of total bytes added.
        offsets[i + 1] = offset_val  # Store cumulative byte count at offsets.

    # ===============
    # Sub step 2.3: Compute lengths and the max length m.
    # ===============
    lengths = offsets[1:] - offsets[:-1]  # lengths[i] = offsets[i+1] - offsets[i].
    m = lengths.max().item()             # The maximum length among all strings.
    total_bytes = offsets[-1].item()     # The total number of bytes across all strings.

    # --------------------------------------------------------------
    # STEP 3: Build 2D index array and gather from base_tensor.
    # --------------------------------------------------------------
    # ===============
    # Sub step 3.1: Create row_ids and col_ids for indexing.
    # ===============
    row_ids = torch.arange(n, dtype=torch.long, device=device).unsqueeze(1).expand(n, m)
    col_ids = torch.arange(m, dtype=torch.long, device=device).unsqueeze(0).expand(n, m)

    # ===============
    # Sub step 3.2: Build 2D indices into a single 1D base_tensor.
    # ===============
    index_2d = offsets[:-1][row_ids] + col_ids
    index_2d.clamp_(0, total_bytes - 1)  # Clamp to valid range for safety.

    # ===============
    # Sub step 3.3: Convert big_data into a torch.Tensor on the correct device.
    # ===============
    # Use numpy.frombuffer() to interpret big_data as uint8, then pass to torch.tensor().
    base_tensor = torch.tensor(
        np.frombuffer(big_data, dtype=np.uint8),
        dtype=torch.uint8,
        device=device
    )

    # ===============
    # Sub step 3.4: Gather bytes to build codes_2D.
    # ===============
    codes_2D = base_tensor[index_2d]

    # --------------------------------------------------------------
    # STEP 4: Pad out-of-bounds codes with zeros.
    # --------------------------------------------------------------
    # ===============
    # Sub step 4.1: Determine which positions are outside actual lengths.
    # ===============
    mask = col_ids >= lengths[row_ids]  # True if the current col_id >= string length.

    # ===============
    # Sub step 4.2: Zero-out those positions.
    # ===============
    codes_2D[mask] = 0

    return codes_2D, lengths


def tensor_to_list(
    codes_2D: torch.Tensor,
    lengths: torch.Tensor,
    encoding: str = "utf-8"
) -> List[str]:
    """
    Description:
        Decodes a 2D padded codes tensor and associated 1D lengths tensor back into
        a list of Python strings.

    Mathematical details:
        Let n = codes_2D.shape[0].
        For each i in {0, 1, ..., n-1}, we read lengths[i] = L_i.
        We extract codes_2D[i, :L_i], call _tensor_slice_to_bytes, then decode to string.

    Parameters:
        codes_2D (torch.Tensor): A 2D uint8 tensor containing padded byte codes.
        lengths (torch.Tensor): A 1D tensor of shape (n) containing string lengths.
        encoding (str): The text encoding used to decode the bytes.

    Returns:
        results (List[str]): A list of decoded Python strings.
    """

    # --------------------------------------------------------------
    # STEP 1: Handle empty input.
    # --------------------------------------------------------------
    # If codes_2D is empty, return an empty list.
    n = codes_2D.shape[0]
    if n == 0:
        return []

    # --------------------------------------------------------------
    # STEP 2: Reconstruct strings from codes_2D using lengths.
    # --------------------------------------------------------------
    results = []
    for i in range(n):
        # ===============
        # Sub step 2.1: Retrieve the length of the i-th string.
        # ===============
        length_i = lengths[i].item()

        # ===============
        # Sub step 2.2: Slice codes_2D for the i-th string.
        # ===============
        row_slice = codes_2D[i, :length_i]

        # ===============
        # Sub step 2.3: Convert to bytes using _tensor_slice_to_bytes.
        # ===============
        row_bytes = _tensor_slice_to_bytes(row_slice)

        # ===============
        # Sub step 2.4: Decode bytes to string.
        # ===============
        s = row_bytes.decode(encoding)
        results.append(s)

    return results


def string_to_tensor(
    s: str,
    encoding: str = "utf-8",
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    """
    Description:
        Converts a single Python string into a 1D torch.Tensor of dtype=torch.uint8,
        which can optionally reside on a specified device (CPU or GPU).

    Mathematical details:
        Let the length of the string be L = len(s.encode(encoding)).
        We form a tensor t of shape (L), where:
            t[j] = s.encode(encoding)[j]
        for j in {0, 1, ..., L-1}.

    Parameters:
        s (str): The string to be converted to a tensor.
        encoding (str): The text encoding to use for the conversion. Defaults to "utf-8".
        device (torch.device): The device on which to create the tensor.

    Returns:
        base_tensor (torch.Tensor): A 1D uint8 tensor representing the encoded string.
    """

    # --------------------------------------------------------------
    # STEP 1: Encode the string to bytes in Python.
    # --------------------------------------------------------------
    b = s.encode(encoding)

    # --------------------------------------------------------------
    # STEP 2: Handle empty string.
    # --------------------------------------------------------------
    if len(b) == 0:
        return torch.empty((0,), dtype=torch.uint8, device=device)

    # --------------------------------------------------------------
    # STEP 3: Convert to NumPy then to torch.Tensor.
    # --------------------------------------------------------------
    # ===============
    # Sub step 3.1: Use np.frombuffer to interpret the Python bytes as uint8.
    # ===============
    # ===============
    # Sub step 3.2: Create the final tensor on the specified device.
    # ===============
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
    Description:
        Converts a 1D torch.Tensor of dtype=torch.uint8 into a Python string.

    Mathematical details:
        Let L = codes_1D.shape[0].
        We read the bytes via _tensor_slice_to_bytes, obtaining a Python bytes object b.
        We then do b.decode(encoding) to obtain the final string.

    Parameters:
        codes_1D (torch.Tensor): A 1D torch.Tensor representing encoded bytes.
        encoding (str): The text encoding to use for decoding. Defaults to "utf-8".

    Returns:
        decoded_string (str): The decoded Python string.
    """

    # --------------------------------------------------------------
    # STEP 1: Handle empty tensor.
    # --------------------------------------------------------------
    if codes_1D.numel() == 0:
        return ""

    # --------------------------------------------------------------
    # STEP 2: Convert tensor to bytes, then decode.
    # --------------------------------------------------------------
    data_bytes = _tensor_slice_to_bytes(codes_1D)
    return data_bytes.decode(encoding)


def patch_functional() -> None:
    """
    Description:
        Monkey-patches the torch.nn.functional module to add four utility functions:
        list_to_tensor, tensor_to_list, string_to_tensor, and tensor_to_string.

    Mathematical details:
        This is not strictly a mathematical function, but rather an API patch.
        F.list_to_tensor = list_to_tensor
        F.tensor_to_list = tensor_to_list
        F.string_to_tensor = string_to_tensor
        F.tensor_to_string = tensor_to_string

    Parameters:
        None

    Returns:
        None
    """

    # --------------------------------------------------------------
    # STEP 1: Assign each function to torch.nn.functional.
    # --------------------------------------------------------------
    F.list_to_tensor = list_to_tensor    # Patch torch.nn.functional with custom list_to_tensor.
    F.tensor_to_list = tensor_to_list    # Patch torch.nn.functional with custom tensor_to_list.
    F.string_to_tensor = string_to_tensor  # Patch torch.nn.functional with custom string_to_tensor.
    F.tensor_to_string = tensor_to_string  # Patch torch.nn.functional with custom tensor_to_string.
