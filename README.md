# FQL-AGLEACH — simulator and results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23150738.svg)](https://doi.org/10.5281/zenodo.23150738)

Source code and per-topology results for the article

> M. Ridwan, T. Wahyono, I. Sembiring, R. Darmastuti,
> *FQL-AGLEACH: An Adaptive Grid LEACH Protocol with Fuzzy Q-Learning for Energy-Efficient Wireless Sensor Networks* (submitted, 2026).

FQL-AGLEACH combines a tile-aligned adaptive grid, whose number of cells follows an optimal-cluster-count estimate, with a 27-rule Glorennec–Jouffe fuzzy Q-learning agent hosted at the base station. In every grid cell, the agent chooses the cluster head among the energy-eligible nodes (residual energy at or above the 75th percentile of the cell). The repository contains the simulator for FQL-AGLEACH and the four baselines (LEACH, Fuzzy-LEACH, Grid-LEACH, AGLEACH), plus the scripts that produce every table and figure of the article.

## Contents

| Path | Description |
|---|---|
| `code/fqlsim.py` | Numba-compiled round-based simulator: radio model, control-plane model, all protocols and the ablation variants |
| `code/runlib.py` | Helpers to run one simulation and compute the lifetime metrics |
| `code/experiments.py` | Scenarios S1–S4, test seeds 100–129; main comparison, ablation and sensitivity runs |
| `code/behaviour.py` | Characteristics of the selected cluster heads (trained vs. frozen agent) |
| `code/analysis.py` | Wilcoxon signed-rank tests with Holm correction; LaTeX tables |
| `code/senstab.py` | Sensitivity table |
| `code/figures.py`, `code/schem.py` | Result figures, network snapshot and flowchart |
| `results/hasil_per_topologi.csv` | Per-topology results of the main comparison (4 scenarios × 5 protocols × 30 topologies) |
| `results/tab_*.tex` | Tables as included in the article |

## Scenarios

| Scenario | Nodes | Deployment | Base station | Rounds |
|---|---|---|---|---|
| S1 | 100 | uniform | (50, 50) | 4000 |
| S2 | 500 | uniform | (50, 50) | 5000 |
| S3 | 100 | clustered (5 hot-spots) | (50, 50) | 4000 |
| S4 | 100 | uniform | (50, 150) | 4000 |

The field is 100 m × 100 m. Every node starts with 1 J and sends one 4000-bit reading per round. The simulator uses the first-order radio model, and control-plane energy is charged to every protocol. Seeds 0–9 were used for development and parameter choices; seeds 100–129 were used for every reported result.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Reproducing the results

Run all commands from the `code/` folder. Each step writes its output to `code/results/` (or `code/figs/` for figures).

```bash
cd code
python experiments.py main    # main comparison  -> results/main.pkl
python experiments.py abl     # ablation         -> results/abl.pkl
python experiments.py sens    # sensitivity      -> results/sens.pkl
python behaviour.py           # CH characteristics -> results/behaviour.pkl
python analysis.py            # statistics and tables -> results/tab_main.tex, tab_sig.tex, tab_abl.tex
python senstab.py             # results/tab_sens.tex
python figures.py             # figs/*.pdf
python schem.py               # figs/fig_network.pdf, figs/fig_flow.pdf
```

On a 2-core 2.8 GHz machine, the whole pipeline takes about 25 minutes. The first call takes a few extra seconds while Numba compiles the simulator.

## Column description of `results/hasil_per_topologi.csv`

| Column | Meaning |
|---|---|
| `scenario` | S1–S4 |
| `protocol` | LEACH, Fuzzy-LEACH, Grid-LEACH, AGLEACH, FQL-AGLEACH |
| `seed` | topology seed (100–129) |
| `FND`, `HND`, `T90`, `LND` | round at which the first node, 50 %, 90 % and all nodes have died |
| `deliv` | readings delivered to the base station over the horizon |
| `ctrl` | control-plane energy as a fraction of all energy spent |
| `jain` | Jain fairness index of residual energy of alive nodes, averaged up to HND |
| `jch` | Jain fairness index of cluster-head duty counts up to HND |
| `reass` | fraction of alive nodes whose cluster head changes between consecutive rounds, averaged up to HND |
| `ch` | mean number of cluster heads per round up to HND |

## License

The code is released under the MIT License. The data in `results/` are released under CC BY 4.0.

## Contact

Mohammad Ridwan — mridwan@unis.ac.id
