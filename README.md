# pyUDLF — gitclone branch (UNDER DEVELOPMENT! Visit the og/bugfix branch)

Originally developed by **Gustavo Rosseto Leticio**, **Lucas Pascotti Valem** and **Daniel Carlos Guimarães Pedronette** (Universidade Estadual Paulista — UNESP, Rio Claro, Brazil).

Bug fixes by **Bionda Rozin**.

> For the full list of changes in this branch, see [`FIXES_pyudlf.md`](FIXES_pyudlf.md).

---

## Overview

pyUDLF is a Python wrapper for [UDLF](https://github.com/UDLF/UDLF) (Unsupervised Distance Learning Framework), a C++ library implementing graph- and rank-based re-ranking methods that refine distance/similarity structures without supervision.

All heavy computation runs in the UDLF binary. pyUDLF handles configuration, execution, and result parsing — no manual `.ini` editing required. If the binary is not found at the configured path, it is downloaded automatically.

> **⚠️ macOS is not supported.** The UDLF binary is only available for Linux and Windows (x86_64).  
> **⚠️ Paths with spaces are not supported.** The UDLF binary does not handle spaces in file paths. Use underscores instead.

---

## Installation

```bash
git clone https://github.com/your-fork/pyUDLF.git
cd pyUDLF
pip install -r requirements.txt
python setup.py install
```

**Requirements:** Python 3.10+, `numpy`, `Pillow`, `requests`.

---

## Tutorial

### 1. Imports and paths

```python
from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType as it

# Optional: set custom paths to the binary and config.
# If not set, pyUDLF uses ~/.pyudlf/ and downloads the binary on first run.
# udlf.setBinaryPath("/opt/udlf/bin/udlf")
# udlf.setConfigPath("/opt/udlf/bin/config.ini")
```

### 2. Configure the input

```python
input_data = it.InputType()

# Method and parameters
input_data.set_method_name("LHRR")
input_data.set_method_parameters("LHRR", k=18, l=1000, t=2)
input_data.set_task("UDL")

# Dataset files — no spaces in paths
input_data.set_input_files("mpeg7/CFD.txt")
input_data.set_input_images_path("mpeg7/original")
input_data.set_lists_file("mpeg7/lists_mpeg7.txt")
input_data.set_classes_file("mpeg7/classes_mpeg7.txt")
input_data.set_dataset_size(1400)

# Output
input_data.set_output_file(True)
input_data.set_output_file_format("RK")   # "RK" or "MATRIX"
input_data.set_output_file_path("./output")
input_data.set_output_log_file_path("./log.txt")

# Effectiveness evaluation
input_data.set_effectiveness_eval(True)
input_data.set_effectiveness_compute_map(True)
```

### 3. Run

```python
output = udlf.run(input_data, get_output=True)
```

### 4. Retrieve results

```python
# Print the full log (MAP, Precision@K, Recall@K, time)
output.print_log()

# Ranked lists as a numpy array (top_k elements per query)
rks = output.get_rks(top_k=100)

# Log as a dictionary
log = output.get_log()
print(log["MAP"]["After"])   # also: "Before", "Gain"

# Distance/similarity matrix — only when output format is "MATRIX"
# matrix = output.get_matrix()
```

---

## Inspecting and modifying parameters

```python
# List all parameters with their current values
input_data.list_param()

# Same, but with comments from config.ini
input_data.list_param_full()

# Info about a specific parameter
input_data.list_param_info("PARAM_LHRR_K")

# Info about all parameters of a method
input_data.list_method_info("LHRR")

# Get/set a parameter by name
input_data.get_param("PARAM_LHRR_K")
input_data.set_param("PARAM_LHRR_K", 25)

# Add a parameter not in the original config
input_data.add_new_parameter("MY_CUSTOM_PARAM", "value")

# Save the current config to a file
input_data.write_config("my_config.ini")
```

> **Note:** `list_param()` shows values from the base `config.ini`, not the values set via `input_data`. Your changes are applied only at runtime when `udlf.run()` generates a temporary config.

---

## Running from an existing config file

You can edit a `config.ini` manually and run it directly:

```python
output = udlf.runWithConfig(
    config_file="/path/to/config.ini",
    get_output=True,
    compute_individual_gain=True,
    depth=100,
)

gain_list = output.get_individual_gain_list(sort=True)
# [(0.12, 4), (0.09, 17), (-0.03, 2), ...]  — (gain, element_index)
```

---

## Visualizing ranked lists

```python
from IPython.display import display

# Ranked list before re-ranking (from input)
img = input_data.show_input_rk(line=0, rk_size=10)
display(img)

# Ranked list after re-ranking (requires images_path set and get_output=True)
img = output.show_rk(line=0, rk_size=10, images_shape=(128, 128))
display(img)

# Save to file
output.save_rk_img(line=0, rk_size=10, img_path="rk_query0.png")
```

Each image in the ranked list is framed by a colored border: **blue** = query, **green** = correct class, **red** = incorrect class.

Example output for query 0 on the MPEG-7 dataset (LHRR, top-10):

![rk_query0](rk_query0.png)

> **Note for Jupyter users:** use `display(img)` from `IPython.display` instead of `img.show()`. The latter attempts to open an external image viewer which may fail depending on your system configuration.

---

## Grid search

Find the best value for a single parameter by MAP:

```python
from pyUDLF.utils import gridSearch

input_data.set_method_name("CPRR")
input_data.set_effectiveness_eval(True)
input_data.set_effectiveness_compute_map(True)

best = gridSearch.find_best_param(
    input_type=input_data,
    method="CPRR",
    param_value="PARAM_CPRR_K",
    list_values=[5, 10, 15, 20, 25, 30],
)
```

---

## Standalone evaluation utilities

These work without running the UDLF binary, useful for evaluating your own ranked lists:

```python
from pyUDLF.utils import evaluation, readData

# Load ranked lists and class labels
rks_before = readData.read_ranked_lists_file_numeric("mpeg7/CFD.txt", top_k=100)
classes = readData.read_classes("mpeg7/lists_mpeg7.txt", "mpeg7/classes_mpeg7.txt")

# Run UDLF and capture output ranked lists
output = udlf.run(input_data, get_output=True)
rks_after = output.get_rks(top_k=100)

# Metrics
map_score, map_per_query = evaluation.compute_map(rks_after, classes, map_depth=100)
recall, recall_per_query = evaluation.compute_recall(rks_after, classes, r_depth=10)
precision, precision_per_query = evaluation.compute_precision(rks_after, classes, p_depth=5)

# Per-element gain between before and after re-ranking
gain_list = evaluation.compute_gain(
    before_rks=rks_before,
    after_rks=rks_after,
    classes_list=classes,
    depth=100,
    measure="MAP",
)
# [(gain_i, index_i), ...]
```

---

## Available methods

| Method | Task | Description |
|--------|------|-------------|
| `NONE` | UDL | Baseline passthrough |
| `CPRR` | UDL | Contextual re-ranking by reciprocal ranks |
| `LHRR` | UDL, FUSION | Log-based hypergraph of ranking references — generally strong |
| `RLSIM` | UDL, FUSION | Ranked list similarity (multiple metrics) |
| `RLRECOM` | UDL | Ranked list recommendation |
| `CONTEXTRR` | UDL, FUSION | Contextual re-ranking with reciprocal references |
| `RECKNNGRAPH` | UDL, FUSION | Reciprocal K-NN graph |
| `RKGRAPH` | UDL, FUSION | Ranked-list graph |
| `CORGRAPH` | UDL, FUSION | Correlation graph |
| `BFSTREE` | UDL | BFS tree |
| `RDPAC` | UDL, FUSION | Reverse K-NN diffusion with positive and adaptive constraints |
| `RFE` | UDL, FUSION | Ranked-list feature embedding |

---

## Compatibility

- **OS:** Linux, Windows. macOS is not supported.
- **Python:** 3.10+
- **Architecture:** x86_64

---

## Citation

If you use this software, please cite the original pyUDLF paper:

```bibtex
@inproceedings{pyUDLF,
    author = {Gustavo Leticio and Lucas Pascotti Valem and Leonardo Tadeu Lopes
              and Daniel Carlos Guimarães Pedronette},
    title = {PyUDLF: A Python Framework for Unsupervised Distance Learning Tasks},
    year = {2023},
    isbn = {9798400701085},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3581783.3613466},
    doi = {10.1145/3581783.3613466},
    booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
    pages = {9680--9684},
    series = {MM '23}
}
```

## License

GPLv2. See [LICENSE](LICENSE).
