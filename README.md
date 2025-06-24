# **Generating Sufficient Querysets for Information Pursuit using Sparse Representative Modeling**

This repository contains the implementation in python using pytorch of the Sparse Modeling Representative Selection algorithm proposed by Elhamifar et al. in "See all by looking at a Few " for the Bachelor's Thesis of **Carl Kemmerich**, supervised by Prof. Dr. Gitta Kutyniok and Stefan Kolek M.Sc. at the Bavarian AI Chair for Mathematical Foundations of Artificial Intelligence at Ludwig-Maximilians-Universität Munich.

---

## Getting Started

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/carlk13/smrs.git
    cd smrs
    ```
2.  **Prepare the environment:**
    ```bash
    uv sync
    ```

---

## Usage
### Running smrs on the cosine similarity matrix of a queryset and cifar10
Run the `main.py` file to select representative queries
```bash
uv run python main.py \
    path/to/your_query_file.txt \
    100 \
    10 \
    --alpha 10.0 \
    --r 0 \
    --max_iterations 200 \
    --delta 0.05 \
    --verbose \
    --run_without_pruning
```

### Running smrs on a standard dataset to test its effectiveness
Run the `run_smrs_on_dataset` script for example with the following command
```bash
uv run python run_smrs_on_dataset.py \
    --dataset cifar10 \
    --split test \
    --filter_class 8 \
    --alpha 10 \
    --r 0 \
    --max_iterations 200 \
    --verbose
    

