"""
This script benchmarks various aspects of string_tensors functionality:
  1) Time to create random lists of strings, or a single random string.
  2) Time to convert those lists or single string to a tensor (CPU/GPU).
  3) Time to convert them back from tensor to python strings (CPU/GPU).
  4) Combined creation -> to_tensor -> from_tensor overhead.
Outputs:
  - Text file with numeric data in 'benchmarks_output/benchmark_results.txt'
  - Multiple PNG plots in 'benchmarks_output' folder.
"""

import os
import random
import string
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import local patch
from torch_string_tensors import patch_functional
patch_functional()

os.makedirs("benchmarks_output", exist_ok=True)

###############################################################################
# Utility to generate random strings
###############################################################################

def generate_random_string(min_len=1, max_len=100, alphabet=None):
    """
    Generate a single random string with length in [min_len, max_len].
    """
    if alphabet is None:
        alphabet = string.ascii_letters + string.digits
    length = random.randint(min_len, max_len)
    return "".join(random.choices(alphabet, k=length))

def generate_random_string_list(num_strings, min_len=1, max_len=100, alphabet=None):
    """
    Generate a list of random strings.
    """
    return [generate_random_string(min_len, max_len, alphabet) for _ in range(num_strings)]

###############################################################################
# Benchmark routines
###############################################################################

def benchmark_creation_list(num_strings, min_len=1, max_len=100, runs=10):
    """
    Measure time to create a list of random strings (without conversion).
    Returns average time in seconds.
    """
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        _ = generate_random_string_list(num_strings, min_len, max_len)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs

def benchmark_creation_single_string(min_len=1, max_len=100, runs=10):
    """
    Measure time to create a single random string.
    Returns average time in seconds.
    """
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        _ = generate_random_string(min_len, max_len)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs

def benchmark_list_to_tensor(strings, device="cpu", runs=10):
    """
    Benchmark only the list_to_tensor (no creation).
    Returns average time in seconds.
    """
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        codes, lengths = F.list_to_tensor(strings)
        if device != "cpu":
            codes = codes.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs

def benchmark_tensor_to_list(codes, lengths, device="cpu", runs=10):
    """
    Benchmark only the tensor_to_list (no creation).
    Returns average time in seconds.
    """
    total_time = 0.0
    codes_device = codes.to(device, non_blocking=True)
    lengths_device = lengths.to(device, non_blocking=True)
    for _ in range(runs):
        start = time.perf_counter()
        _ = F.tensor_to_list(codes_device, lengths_device)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs

def benchmark_single_to_tensor(s, device="cpu", runs=10):
    """
    Benchmark only the string_to_tensor.
    Returns average time in seconds.
    """
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        t = F.string_to_tensor(s)
        if device != "cpu":
            t = t.to(device, non_blocking=True)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs

def benchmark_single_from_tensor(t, device="cpu", runs=10):
    """
    Benchmark only the tensor_to_string.
    Returns average time in seconds.
    """
    t_device = t.to(device, non_blocking=True)
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        _ = F.tensor_to_string(t_device)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs

def benchmark_combined_list(num_strings, min_len=1, max_len=100, device="cpu", runs=10):
    """
    Combined: creation -> list_to_tensor -> tensor_to_list
    Returns average time in seconds.
    """
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        strings = generate_random_string_list(num_strings, min_len, max_len)
        codes, lengths = F.list_to_tensor(strings)
        if device != "cpu":
            codes = codes.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
        _ = F.tensor_to_list(codes, lengths)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs

def benchmark_combined_single(min_len=1, max_len=100, device="cpu", runs=10):
    """
    Combined: creation -> string_to_tensor -> tensor_to_string
    Returns average time in seconds.
    """
    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        s = generate_random_string(min_len, max_len)
        t = F.string_to_tensor(s)
        if device != "cpu":
            t = t.to(device, non_blocking=True)
        _ = F.tensor_to_string(t)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / runs

###############################################################################
# Main run and plotting
###############################################################################

def run_benchmarks_and_plot():
    """
    Runs multiple benchmark scenarios, saves the numeric data to a text file,
    and plots combined CPU/GPU results in 'benchmarks_output' folder.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    scenarios_list = [
        (10,  "N=10"),
        (100, "N=100"),
        (500, "N=500"),
    ]

    runs = 100

    # We'll gather data in a dictionary
    results_data = {
        "list_creation": {},       # Key: device -> list of (label, time)
        "list_to_tensor": {},
        "tensor_to_list": {},
        "combined_list": {},
        "single_creation": {},
        "single_to_tensor": {},
        "single_from_tensor": {},
        "combined_single": {},
    }

    # Because single string scenario doesn't need "N=10" or "N=100",
    # we just use one scenario for single string length ranges:
    single_scenario_label = "SingleString"

    ###########################################################################
    # LIST creation / conversion
    ###########################################################################
    for device in devices:
        results_data["list_creation"][device]    = []
        results_data["list_to_tensor"][device]   = []
        results_data["tensor_to_list"][device]   = []
        results_data["combined_list"][device]    = []

        for (n, n_label) in scenarios_list:
            creation_time = benchmark_creation_list(n, min_len=5, max_len=15, runs=runs)
            results_data["list_creation"][device].append((n_label, creation_time))

            sample_list = generate_random_string_list(n, 5, 15)
            l2t_time = benchmark_list_to_tensor(sample_list, device=device, runs=runs)
            codes, lengths = F.list_to_tensor(sample_list)
            if device != "cpu":
                codes = codes.to(device)
                lengths = lengths.to(device)
            t2l_time = benchmark_tensor_to_list(codes, lengths, device=device, runs=runs)
            combined_time = benchmark_combined_list(n, min_len=5, max_len=15, device=device, runs=runs)

            results_data["list_to_tensor"][device].append((n_label, l2t_time))
            results_data["tensor_to_list"][device].append((n_label, t2l_time))
            results_data["combined_list"][device].append((n_label, combined_time))

    ###########################################################################
    # SINGLE creation / conversion
    ###########################################################################
    for device in devices:
        results_data["single_creation"][device]      = []
        results_data["single_to_tensor"][device]     = []
        results_data["single_from_tensor"][device]   = []
        results_data["combined_single"][device]      = []

        single_creation_time = benchmark_creation_single_string(min_len=5, max_len=15, runs=runs)
        results_data["single_creation"][device].append((single_scenario_label, single_creation_time))

        single_str = generate_random_string(5, 15)
        s2t_time = benchmark_single_to_tensor(single_str, device=device, runs=runs)
        t = F.string_to_tensor(single_str)
        if device != "cpu":
            t = t.to(device)
        stime = benchmark_single_from_tensor(t, device=device, runs=runs)
        combined_time = benchmark_combined_single(min_len=5, max_len=15, device=device, runs=runs)

        results_data["single_to_tensor"][device].append((single_scenario_label, s2t_time))
        results_data["single_from_tensor"][device].append((single_scenario_label, stime))
        results_data["combined_single"][device].append((single_scenario_label, combined_time))

    ###########################################################################
    # Write out the data to a text file
    ###########################################################################
    os.makedirs("benchmarks_output", exist_ok=True)
    txt_output_path = os.path.join("benchmarks_output", "benchmark_results.txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write("Benchmark Results\n")
        f.write("=================\n\n")
        for category in results_data:
            f.write(f"Category: {category}\n")
            for device in results_data[category]:
                f.write(f"  Device: {device}\n")
                for (lbl, timing) in results_data[category][device]:
                    f.write(f"    {lbl}: {timing:.6f} s\n")
            f.write("\n")

    ############################################################################
    # Plot results (CPU and GPU combined in a single figure)
    ############################################################################
    import numpy as np

    def plot_and_save(category_name, title, is_single=False):
        """
        Combine CPU and GPU results in a single bar plot for each category.
        Saves into 'benchmarks_output/category_name.png'.
        """
        # We'll treat "labels" as the unique x-axis from CPU's entry (or GPU).
        # If GPU is absent, we'll just show CPU.
        cpu_entries = results_data[category_name].get("cpu", [])
        gpu_entries = results_data[category_name].get("cuda", [])

        # If there is no CPU data at all (unusual), we just skip
        if not cpu_entries and not gpu_entries:
            return

        # We expect that CPU and GPU have the same .[0], .[1] label/time pairs for the same scenario
        # We'll rely on CPU being the source of labels if it exists. Otherwise GPU.
        if cpu_entries:
            labels = [item[0] for item in cpu_entries]
            times_cpu = [item[1] for item in cpu_entries]
        else:
            labels = [item[0] for item in gpu_entries]
            times_cpu = [0 for _ in gpu_entries]  # if CPU is missing

        # For GPU, we must see if we have it
        if gpu_entries:
            times_gpu = [item[1] for item in gpu_entries]
        else:
            times_gpu = [0 for _ in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6,4))
        bar_cpu = ax.bar(x - width/2, times_cpu, width, label='CPU', color='blue')
        bar_gpu = None
        if gpu_entries:
            bar_gpu = ax.bar(x + width/2, times_gpu, width, label='GPU', color='orange')

        ax.set_title(title)
        ax.set_ylabel("Time (seconds)")

        if not is_single:
            ax.set_xlabel("Number of Strings")
        else:
            ax.set_xlabel("Scenario")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # Put the timing value above each bar
        for rect in bar_cpu:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2.0, height,
                    f"{height:.4f}",
                    ha='center', va='bottom', fontsize=8)
        if bar_gpu:
            for rect in bar_gpu:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2.0, height,
                        f"{height:.4f}",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join("benchmarks_output", f"{category_name}.png"), bbox_inches="tight")
        plt.close(fig)

    # List-related categories
    plot_and_save("list_creation",    "List Creation")
    plot_and_save("list_to_tensor",   "list_to_tensor")
    plot_and_save("tensor_to_list",   "tensor_to_list")
    plot_and_save("combined_list",    "Combined: creation->list_to_tensor->tensor_to_list")

    # Single-related categories
    plot_and_save("single_creation",      "Single String Creation", is_single=True)
    plot_and_save("single_to_tensor",     "string_to_tensor",       is_single=True)
    plot_and_save("single_from_tensor",   "tensor_to_string",       is_single=True)
    plot_and_save("combined_single",      "Combined Single: creation->string_to_tensor->tensor_to_string", is_single=True)
    
    print("Benchmarking complete. Results written to:")
    print(f"  {txt_output_path}")
    print("Plots saved in 'benchmarks_output' folder.")

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    run_benchmarks_and_plot()
