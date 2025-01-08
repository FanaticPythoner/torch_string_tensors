import unittest
import random
import string
import torch
import torch.nn.functional as F

from string_tensors import list_to_tensor, tensor_to_list, string_to_tensor, tensor_to_string, patch_functional

patch_functional()

class TestFunctionalExtensions(unittest.TestCase):
    """
    Comprehensive unit tests for the functional APIs:
    list_to_tensor, tensor_to_list, string_to_tensor, and tensor_to_string.
    """

    def test_empty_list_to_tensor(self) -> None:
        """
        Test that an empty list of strings returns zero-shaped tensors.
        """
        codes, lengths = F.list_to_tensor([])
        self.assertEqual(codes.shape, (0, 0))
        self.assertEqual(lengths.shape, (0,))
        recovered = F.tensor_to_list(codes, lengths)
        self.assertEqual(recovered, [])

    def test_single_short_string(self) -> None:
        """
        Test a single short string for list_to_tensor and tensor_to_list.
        """
        input_strings = ["Hello"]
        codes, lengths = F.list_to_tensor(input_strings)
        self.assertEqual(codes.shape, (1, 5))
        self.assertEqual(lengths.tolist(), [5])
        output_strings = F.tensor_to_list(codes, lengths)
        self.assertEqual(output_strings, input_strings)

    def test_multiple_strings_various_lengths(self) -> None:
        """
        Test multiple strings of varying lengths.
        """
        input_strings = ["alpha", "b", "gamma123", "delta!@#"]
        codes, lengths = F.list_to_tensor(input_strings)
        max_len = max(len(s) for s in input_strings)
        self.assertEqual(codes.shape, (len(input_strings), max_len))
        self.assertEqual(len(lengths), len(input_strings))
        recovered = F.tensor_to_list(codes, lengths)
        self.assertEqual(recovered, input_strings)

    def test_unicode_strings(self) -> None:
        """
        Test strings with Unicode characters.
        """
        input_strings = ["こんにちは", "école", "🙂😇🤖", "𝕬𝖑𝖕𝖍𝖆"]
        codes, lengths = F.list_to_tensor(input_strings, encoding="utf-8")
        recovered = F.tensor_to_list(codes, lengths, encoding="utf-8")
        self.assertEqual(recovered, input_strings)

    def test_random_strings(self) -> None:
        """
        Random ASCII strings of various lengths, checking correctness.
        """
        random_strings = []
        for _ in range(50):
            size = random.randint(1, 200)
            s = "".join(random.choices(string.ascii_letters + string.digits, k=size))
            random_strings.append(s)

        codes, lengths = F.list_to_tensor(random_strings)
        recovered = F.tensor_to_list(codes, lengths)
        self.assertEqual(random_strings, recovered)

    def test_string_to_tensor_and_back(self) -> None:
        """
        Test single string -> tensor -> string round trip.
        """
        s = "Hello World! 😎"
        t = F.string_to_tensor(s)
        self.assertEqual(t.dtype, torch.uint8)
        recovered = F.tensor_to_string(t)
        self.assertEqual(s, recovered)

    def test_string_empty(self) -> None:
        """
        Test an empty single string round trip.
        """
        s = ""
        t = F.string_to_tensor(s)
        self.assertEqual(t.numel(), 0)
        out = F.tensor_to_string(t)
        self.assertEqual(out, "")

    def test_tensors_on_gpu(self) -> None:
        """
        Test codes on GPU to ensure fallback to CPU extraction is correct.
        """
        if torch.cuda.is_available():
            # Test list-based conversion
            strings = ["GPU test", "Café", "Data🚀"]
            codes, lengths = F.list_to_tensor(strings)
            codes_gpu = codes.cuda()
            lengths_gpu = lengths.cuda()
            recovered = F.tensor_to_list(codes_gpu, lengths_gpu)
            self.assertEqual(strings, recovered)

            # Test single string conversion
            single = "SingleGPU"
            single_tensor = F.string_to_tensor(single).cuda()
            out_single = F.tensor_to_string(single_tensor)
            self.assertEqual(single, out_single)
        else:
            # If no GPU, just skip
            self.skipTest("CUDA not available. Skipping GPU test.")

    def test_bfloat16_scenario(self) -> None:
        """
        Test scenario with bfloat16 data. This is contrived for strings,
        but we verify the upcasting logic in _tensor_slice_to_bytes.
        """
        s = "".join(random.choices(string.ascii_lowercase, k=3))
        data = F.string_to_tensor(s)
        data_bf16 = data.to(torch.bfloat16)
        recovered = F.tensor_to_string(data_bf16, encoding="ascii")
        self.assertEqual(s, recovered)


if __name__ == "__main__":
    unittest.main()
