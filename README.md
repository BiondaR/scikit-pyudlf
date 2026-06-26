# pyUDLF — gitclone branch

Originally developed by **Gustavo Rosseto Leticio**, **Lucas Pascotti Valem** and **Daniel Carlos Guimarães Pedronette** (Universidade Estadual Paulista — UNESP, Rio Claro, Brazil).

Bug fixes and extensions by **Bionda Rozin**.

> This branch extends the [og/bugfixes branch](https://github.com/BiondaR/scikit-pyudlf/tree/og/bugfixes) with support for building the UDLF binary from source. If you just want to use pyUDLF normally, the bugfix branch is the right place — it downloads a pre-compiled binary automatically and requires no additional dependencies.

---

## Overview

pyUDLF is a Python wrapper for [UDLF](https://github.com/UDLF/UDLF) (Unsupervised Distance Learning Framework), a C++ library implementing graph- and rank-based re-ranking methods that refine distance/similarity structures without supervision.

All heavy computation runs in the UDLF binary. pyUDLF handles configuration, execution, and result parsing — no manual `.ini` editing required.

This branch adds `build_udlf_from_source()`, which clones the UDLF repository and compiles the binary locally. This gives you access to two builds:

- **`master`** — same codebase as the pre-compiled binary available in the bugfix branch
- **`openmp`** — compiled with `-fopenmp`, enabling multi-threaded parallel execution and faster runtimes on multi-core machines

> **⚠️ Paths with spaces are not supported.** The UDLF binary does not handle spaces in file paths. Use underscores instead.

---

## When to use this branch

Use `build_udlf_from_source()` if:
- you want the **OpenMP build** for faster execution
- you are on **macOS**, where no pre-compiled binary is available (`master` branch only)
- you want to compile from source for any other reason (custom flags, auditing, etc.)

Otherwise, the bugfix branch handles everything automatically.

---

## Additional requirements

Beyond the base requirements (`numpy`, `Pillow`, `requests`), building from source requires:

- `git` — to clone the UDLF repository
- `g++` with C++14 support — to compile

**Linux:**
```bash
sudo apt install git g++
```

**macOS:**
```bash
xcode-select --install   # installs git and clang++ (aliased as g++)
```

**Windows:** not supported by `build_udlf_from_source()`. The pre-compiled binary is downloaded automatically via the standard flow — no compilation needed.

---

## Installation

```bash
git clone https://github.com/BiondaR/scikit-pyudlf.git
cd scikit-pyudlf
git checkout og/gitclone
pip install -r requirements.txt
python setup.py install
```

---

## Building the UDLF binary

```python
from pyUDLF import run_calls as udlf

# Build master branch (same as pre-compiled binary)
udlf.build_udlf_from_source('/path/to/install', branch='master')

# Build OpenMP branch (parallel execution, faster on multi-core)
udlf.build_udlf_from_source('/path/to/install', branch='openmp')
```

The function clones the UDLF repository, compiles it with `make`, copies the binary and `config.ini` to `install_path/bin/`, and removes the cloned repository. Nothing is left behind except the compiled binary.

Example output for `branch='openmp'`:

```
[INFO] Clonando UDLF branch 'openmp'...
[INFO] Compilando...
g++ -I./src -std=gnu++14 -O3 -fopenmp -c src/Core/Main.cpp -o obj/Main.o
...
g++ -I./src -std=gnu++14 -O3 -fopenmp ... -o bin/udlf
[INFO] Binário instalado em: /path/to/install/bin/udlf
[INFO] Repositório removido: /path/to/install/UDLF
```

After building, point pyUDLF to the new binary:

```python
udlf.setBinaryPath('/path/to/install/bin/udlf')
udlf.setConfigPath('/path/to/install/bin/config.ini')
```

---

## OS compatibility

| OS | Pre-compiled binary | `master` build | `openmp` build |
|----|---------------------|----------------|----------------|
| Linux | ✅ | ✅ | ✅ |
| Windows | ✅ | ❌ | ❌ |
| macOS | ❌ | ✅ | ❌ |

macOS users must build from source. The `openmp` branch is not supported on macOS because Apple's `clang++` does not support `-fopenmp` natively. Use `branch='master'` instead.

---

## Tutorial

### 1. Imports and paths

```python
from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType as it

# Build from source and point pyUDLF to the binary
udlf.build_udlf_from_source('/path/to/install', branch='openmp')
udlf.setBinaryPath('/path/to/install/bin/udlf')
udlf.setConfigPath('/path/to/install/bin/config.ini')
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

```python
from pyUDLF.utils import evaluation, readData

rks_before = readData.read_ranked_lists_file_numeric("mpeg7/CFD.txt", top_k=100)
classes = readData.read_classes("mpeg7/lists_mpeg7.txt", "mpeg7/classes_mpeg7.txt")

output = udlf.run(input_data, get_output=True)
rks_after = output.get_rks(top_k=100)

map_score, map_per_query = evaluation.compute_map(rks_after, classes, map_depth=100)
recall, recall_per_query = evaluation.compute_recall(rks_after, classes, r_depth=10)
precision, precision_per_query = evaluation.compute_precision(rks_after, classes, p_depth=5)

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
