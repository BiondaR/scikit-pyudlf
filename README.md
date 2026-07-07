<div align="center">

# scikit-pyUDLF

**A friendlier fork of pyUDLF — sklearn API, source builds, and bug fixes**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: GPLv2](https://img.shields.io/badge/license-GPLv2-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()

</div>

Originally developed by **Gustavo Rosseto Leticio**, **Lucas Pascotti Valem** and **Daniel Carlos Guimarães Pedronette** (Universidade Estadual Paulista — UNESP, Rio Claro, Brazil).

Bug fixes and extensions by **Bionda Rozin**.

---

## Overview

pyUDLF is a Python wrapper for [UDLF](https://github.com/UDLF/UDLF) (Unsupervised Distance Learning Framework), a C++ library implementing graph- and rank-based re-ranking methods that refine distance/similarity structures without supervision.

This is the **main branch** — the most complete version of pyUDLF. It includes all bug fixes and the source-build support from the other branches, and adds a sklearn-style `UDLF` class that:

- accepts **raw feature matrices** directly (no need to pre-compute ranked lists or distance files)
- exposes **evaluation metrics** (`precision@k`, `recall@k`, `ndcg@k`, `mAP`, `MRR`, `jaccard overlap`) directly on the fitted model
- supports **comparing** multiple methods on the same dataset in one call
- follows the `fit` / `transform` / `fit_transform` interface, making it compatible with sklearn pipelines

The low-level API (`run_calls` + `InputType`) is still fully available for cases that need fine-grained control.

> **⚠️ Paths with spaces are not supported.** The UDLF binary does not handle spaces in file paths. Use underscores instead.

---

## Other branches

| Branch | What it adds |
|--------|-------------|
| `og/bugfixes` | Bug fixes over the original pyUDLF. Downloads a pre-compiled binary automatically. Starting point for everything else. |
| `og/gitclone` | Extends `og/bugfixes` with `build_udlf_from_source()` — clone + compile from the UDLF repo. Enables the OpenMP build and macOS support. |
| `main` | This branch. Adds the `UDLF` sklearn-style class on top of `og/gitclone`. |

---

## Installation

```bash
git clone https://github.com/BiondaR/scikit-pyudlf.git
cd scikit-pyudlf
pip install -r requirements.txt
python setup.py install
```

**Requirements:** Python 3.10+, `numpy`, `scikit-learn`, `pandas`, `Pillow`, `requests`.

---

## Building the UDLF binary from source

By default, pyUDLF downloads a pre-compiled binary automatically (Linux and Windows only). You can also build it yourself, which gives you access to two versions:

- **`master`** — same codebase as the pre-compiled binary
- **`openmp`** — compiled with `-fopenmp`, enabling multi-threaded parallel execution and faster runtimes on multi-core machines

Use `build_udlf_from_source()` if:
- you want the **OpenMP build** for faster execution
- you are on **macOS**, where no pre-compiled binary is available (`master` branch only)
- you want to compile from source for any other reason (custom flags, auditing, etc.)

### Additional requirements

Beyond the base requirements, building from source requires:

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

### Usage

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

### OS compatibility for source builds

| OS | Pre-compiled binary | `master` build | `openmp` build |
|----|---------------------|----------------|----------------|
| Linux | ✅ | ✅ | ✅ |
| Windows | ✅ | ❌ | ❌ |
| macOS | ❌ | ✅ | ❌ |

macOS users must build from source. The `openmp` branch is not supported on macOS because Apple's `clang++` does not support `-fopenmp` natively. Use `branch='master'` instead.

---

## Quick start — `UDLF` class

### 1. Imports and paths

```python
from pyUDLF.api import UDLF
import numpy as np

RKS     = "mpeg7/CFD.txt"
LISTS   = "mpeg7/lists_mpeg7.txt"
CLASSES = "mpeg7/classes_mpeg7.txt"
IMAGES  = "mpeg7/original"
```

### 2. `__repr__` — inspecting the object

```python
model = UDLF(task="UDL")
model.lhrr(L=1000, K=18, T=2)
print(model)
# UDLF(task='UDL', method='LHRR', persist=False, status='não executado')
```

### 3. Fit with `rks_path`

```python
model = UDLF(
    task="UDL",
    rks_path=RKS,
    list_path=LISTS,
    classes_path=CLASSES,
)
model.lhrr(L=1000, K=18, T=2)
model.fit()

print(model)
# UDLF(task='UDL', method='LHRR', persist=False, status='fitted')
```

### 4. Fit with a feature matrix

```python
X = np.random.rand(1400, 512)
y = model.get_labels_from_classes_file()

model2 = UDLF(task="UDL")
model2.lhrr(L=1000, K=18, T=2)
model2.fit(X, y)
```

Passing a pre-computed distance matrix:

```python
from sklearn.metrics.pairwise import euclidean_distances

dist = euclidean_distances(X, X)

model = UDLF(task="UDL")
model.lhrr(L=1000, K=18, T=2)
model.fit(dist, y, distance='precomputed')
```

### 5. `fit_transform`

```python
model = UDLF(task="UDL")
model.lhrr(L=1000, K=18, T=2)
dist_matrix = model.fit_transform(X, y)   # (1400, 1400) float array
```

### 6. `get_metrics` — UDLF built-in log

```python
model = UDLF(task="UDL", rks_path=RKS, list_path=LISTS, classes_path=CLASSES)
model.lhrr(L=1000, K=18, T=2)
model.fit()

# dict with Before/After/Gain
print(model.get_metrics())

# only the After values
print(model.get_metrics(mode='After'))

# as a DataFrame
df = model.get_metrics(output_type='df')
```

### 7. Supervised metrics

```python
y = model.get_labels_from_classes_file()

print(model.precision_at_k(y=y, k=10))
print(model.recall_at_k(y=y, k=10))
print(model.f1_score_at_k(y=y, k=10))
print(model.average_precision_at_k(y=y, k=10))   # mAP when mode='global'
print(model.reciprocal_rank(y=y))                 # MRR when mode='global'
print(model.ndcg_at_k(y=y, k=10))
```

### 8. Aggregation modes (`global` / `class` / `index`)

```python
# global — float
print(model.precision_at_k(y=y, k=10, mode='global'))

# per class — dict {class: score}
print(model.precision_at_k(y=y, k=10, mode='class'))
# {'apple': 0.91, 'bat': 0.87, ...}

# per query — dict {index: score}
print(model.precision_at_k(y=y, k=10, mode='index'))
# {0: 0.9, 1: 0.8, ...}
```

Works with all supervised metrics (`recall_at_k`, `f1_score_at_k`, `ndcg_at_k`, `average_precision_at_k`, `reciprocal_rank`).

### 9. Unsupervised metric — `jaccard_overlap_at_k`

Measures how much the top-K neighborhood changed before and after re-ranking. Requires no labels.

```python
# global
overlap = model.jaccard_overlap_at_k(k=10)
print(overlap)   # float in [0, 1] — 1.0 = nothing changed, 0.0 = fully replaced

# per class (requires y)
overlap_class = model.jaccard_overlap_at_k(y=y, k=10, mode='class')
```

### 10. Persisting outputs

```python
model = UDLF(
    task="UDL",
    rks_path=RKS,
    list_path=LISTS,
    classes_path=CLASSES,
    persist=True,
    log_path="outputs/log.txt",
    output_path="outputs/output",   # extension added automatically
)
model.lhrr(L=1000, K=18, T=2)
model.fit()
# outputs/output.txt and outputs/log.txt are saved to disk
```

### 11. Fusion

```python
FUSION_PATHS = [
    "mpeg7/CFD.txt", "mpeg7/AIR.txt", "mpeg7/ASC.txt",
    "mpeg7/BAS.txt", "mpeg7/IDSC.txt", "mpeg7/SS.txt",
]

model = UDLF(
    task="FUSION",
    rks_path=FUSION_PATHS,
    list_path=LISTS,
    classes_path=CLASSES,
    persist=True,
    log_path="outputs/log_fusion.txt",
    output_path="outputs/output_fusion",
)
model.lhrr(L=1000, K=18, T=2)
model.fit()
```

### 12. `compare`

Re-fits all given models on the exact same data and returns a comparison DataFrame:

```python
model_lhrr = UDLF(task="UDL", rks_path=RKS, list_path=LISTS, classes_path=CLASSES)
model_lhrr.lhrr(L=1000, K=18, T=2)

model_cprr = UDLF(task="UDL", rks_path=RKS, list_path=LISTS, classes_path=CLASSES)
model_cprr.cprr(L=400, K=20, T=2)

results = UDLF.compare({
    "LHRR(K=18)": model_lhrr,
    "CPRR(K=20)": model_cprr,
}, k=10)
results
```

It also works with unfitted models plus `X`/`y`:

```python
model_lhrr = UDLF(task="UDL")
model_lhrr.lhrr(L=1000, K=18, T=2)

model_cprr = UDLF(task="UDL")
model_cprr.cprr(L=400, K=20, T=2)

results = UDLF.compare({
    "LHRR": model_lhrr,
    "CPRR": model_cprr,
}, X=X, y=y, k=10)
results
```

---

## Low-level API — UDL

```python
from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType as it

input_data = it.InputType()
input_data.set_method_name("LHRR")
input_data.set_method_parameters("LHRR", k=18, l=1000, t=2)
input_data.set_task("UDL")
input_data.set_input_files("mpeg7/CFD.txt")
input_data.set_input_images_path("mpeg7/original")
input_data.set_lists_file("mpeg7/lists_mpeg7.txt")
input_data.set_classes_file("mpeg7/classes_mpeg7.txt")
input_data.set_dataset_size(1400)
input_data.set_output_file(True)
input_data.set_output_file_format("RK")   # or "MATRIX"
input_data.set_output_rk_format("NUM")
input_data.set_output_file_path("./output")
input_data.set_output_log_file_path("./log.txt")
input_data.set_effectiveness_eval(True)
input_data.set_effectiveness_compute_map(True)

output = udlf.run(input_data, get_output=True)
output.print_log()
rks = output.get_rks(top_k=100)
log = output.get_log()
print(log["MAP"]["After"])

# matrix = output.get_matrix()  — use with set_output_file_format("MATRIX")
```

---

## Low-level API — FUSION

```python
input_data = it.InputType()
input_data.set_method_name("LHRR")
input_data.set_method_parameters("LHRR", k=18, l=1000, t=2)
input_data.set_task("FUSION")
input_data.set_input_files([
    "mpeg7/CFD.txt", "mpeg7/AIR.txt", "mpeg7/ASC.txt",
    "mpeg7/BAS.txt", "mpeg7/IDSC.txt", "mpeg7/SS.txt"
])
input_data.set_input_images_path("mpeg7/original")
input_data.set_lists_file("mpeg7/lists_mpeg7.txt")
input_data.set_classes_file("mpeg7/classes_mpeg7.txt")
input_data.set_dataset_size(1400)
input_data.set_output_file(True)
input_data.set_output_file_format("RK")
input_data.set_output_rk_format("NUM")
input_data.set_output_file_path("./output")
input_data.set_output_log_file_path("./log.txt")
input_data.set_effectiveness_eval(True)
input_data.set_effectiveness_compute_map(True)

output = udlf.run(input_data, get_output=True)
output.print_log()
log = output.get_log()
print(log)
rks = output.get_rks(top_k=100)

# matrix = output.get_matrix()  — use with set_output_file_format("MATRIX")
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

## Methods and parameters

### [NONE](https://github.com/UDLF/UDLF/wiki/Methods) — baseline passthrough
`model.set_none(L=1400)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 1400 | Ranked list size |

---

### [CPRR](http://dx.doi.org/10.1109/SIBGRAPI.2016.042)
`model.cprr(L=400, K=20, T=2)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 400 | Ranked list size |
| `K` | 20 | Neighborhood size |
| `T` | 2 | Number of iterations |

---

### [LHRR](http://doi.org/10.1109/TIP.2019.2920526)
`model.lhrr(L=1400, K=18, T=2)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 1400 | Ranked list size |
| `K` | 18 | Neighborhood size |
| `T` | 2 | Number of iterations |

---

### [RL-Sim*](http://dx.doi.org/10.1145/2671188.2749335)
`model.rlsim(TOPK=15, CK=700, T=3, METRIC='INTERSECTION')`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOPK` | 15 | Top-K neighbors for similarity |
| `CK` | 700 | Context size |
| `T` | 3 | Number of iterations |
| `METRIC` | `'INTERSECTION'` | `INTERSECTION`, `RBO`, `KENDALL_TAU`, `SPEARMAN`, `GOODMAN`, `JACCARD`, `JACCARD_K`, `KENDALL_TAU_W` |

---

### [RL-Recom](http://dx.doi.org/10.1145/2671188.2749336) *(UDL only)*
`model.rlrecom(L=400, K=8, EPS=0.0125, LAMBDA=2.0)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 400 | Ranked list size |
| `K` | 8 | Neighborhood size |
| `EPS` | 0.0125 | Convergence threshold |
| `LAMBDA` | 2.0 | Regularization weight |

---

### [ContextRR](http://dl.acm.org/citation.cfm?id=1948207.1948291)
`model.contextrr(L=25, K=7, T=5, NBYK=1, OPTIMIZATIONS=True)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 25 | Ranked list size |
| `K` | 7 | Neighborhood size |
| `T` | 5 | Number of iterations |
| `NBYK` | 1 | N/K ratio |
| `OPTIMIZATIONS` | `True` | Enable speed optimizations |

---

### [ReckNNGraph](http://dx.doi.org/10.1016/j.imavis.2013.12.009)
`model.recknngraph(L=200, K=15, EPS=0.0125)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 200 | Ranked list size |
| `K` | 15 | Neighborhood size |
| `EPS` | 0.0125 | Convergence threshold |

---

### [Rk Graph Dist.](http://dx.doi.org/10.1016/j.patrec.2016.05.021)
`model.rkgraph(L=700, K=20, T=1, P=0.95)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 700 | Ranked list size |
| `K` | 20 | Neighborhood size |
| `T` | 1 | Number of iterations |
| `P` | 0.95 | RBO probability threshold |

---

### [Correlation Graph](http://dx.doi.org/10.1016/j.neucom.2016.03.081)
`model.corgraph(L=200, K=25, TH_START=0.35, TH_END=1.0, TH_INC=0.005, CORRELATION='PEARSON')`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 200 | Ranked list size |
| `K` | 25 | Neighborhood size |
| `TH_START` | 0.35 | Threshold start |
| `TH_END` | 1.0 | Threshold end |
| `TH_INC` | 0.005 | Threshold increment |
| `CORRELATION` | `'PEARSON'` | `PEARSON` or `RBO` |

---

### [BFSTree](https://doi.org/10.1016/j.patcog.2020.107666) *(UDL only)*
`model.bfstree(L=1400, K=20, CORRELATION='RBO')`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 1400 | Ranked list size |
| `K` | 20 | Neighborhood size |
| `CORRELATION` | `'RBO'` | Only `RBO` currently supported |

---

### [RDPAC](https://doi.org/10.3390/jimaging7030049)
`model.rdpac(L=400, L_MULT=2, P=0.60, PL=0.99, K_START=1, K_INC=1, K=15)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 400 | Ranked list size |
| `L_MULT` | 2 | L multiplier |
| `P` | 0.60 | Probability threshold |
| `PL` | 0.99 | Probability lower bound |
| `K_START` | 1 | K range start |
| `K_INC` | 1 | K range increment |
| `K` | 15 | K range end |

---

### [RFE](https://doi.org/10.1109/TIP.2023.3268868)
`model.rfe(L=400, K=20, T=2, PA=0.1, TH_CC=0, RK_BY_EMB=False, EXPORT_EMBS=False, PERF_CCS=True, EMB_PATH='embeddings.txt', CCS_PATH='ccs.txt')`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `L` | 400 | Ranked list size |
| `K` | 20 | Neighborhood size |
| `T` | 2 | Number of iterations |
| `PA` | 0.1 | Positive-pair acceptance rate |
| `TH_CC` | 0 | Connected components threshold |
| `RK_BY_EMB` | `False` | Re-rank by embedding distance |
| `EXPORT_EMBS` | `False` | Export learned embeddings |
| `PERF_CCS` | `True` | Perform connected components |
| `EMB_PATH` | `'embeddings.txt'` | Embeddings output path |
| `CCS_PATH` | `'ccs.txt'` | CCS output path |

---

## Method compatibility

| Method | UDL | FUSION |
|--------|-----|--------|
| NONE | ✅ | ❌ |
| CPRR | ✅ | ✅ |
| LHRR | ✅ | ✅ |
| RL-Sim* | ✅ | ✅ |
| RL-Recom | ✅ | ❌ |
| ContextRR | ✅ | ✅ |
| ReckNNGraph | ✅ | ✅ |
| Rk Graph Dist. | ✅ | ✅ |
| Correlation Graph | ✅ | ✅ |
| BFSTree | ✅ | ❌ |
| RDPAC | ✅ | ✅ |
| RFE | ✅ | ✅ |

---

## Compatibility

| | |
|---|---|
| **OS** | Linux, Windows (pre-compiled binary). macOS supported via `build_udlf_from_source()`. |
| **Python** | 3.10+ |
| **Architecture** | x86_64 |

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

## Acknowledgments

The authors are grateful to São Paulo Research Foundation – FAPESP (grants 2013/08645-0, 2014/04220-8, 2018/15597-6) and Brazilian National Council for Scientific and Technological Development – CNPq (grants 309439/2020-5 and 422667/2021-8).

## Contact
**Gustavo Rosseto Leticio**: `gustavo.leticio@gmail.com` or `gustavo.leticio@unesp.br`

**Lucas Pascotti Valem**: `lucas@icmc.usp.br`

**Daniel Carlos Guimarães Pedronette**: `daniel.pedronette@unesp.br`

**Bionda Rozin**: `bionda.rozin@unesp.br`

## License

GPLv2. See [LICENSE](LICENSE).
