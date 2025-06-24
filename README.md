# Sparse Modeling Representative Selection

**Generating Sufficient Querysets for Information Pursuit using Sparse Representative Modeling**

This repository contains the implementation for the Bachelor's Thesis of **Carl Kemmerich**, supervised by Prof. Dr. Gitta Kutyniok and Stefan Kolek M.Sc. at the Bavarian AI Chair for Mathematical Foundations of Artificial Intelligence.

---

## Getting Started

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd smrs
    ```
2.  **Prepare the environment:**
    ```bash
    uv sync
    ```

---

## Usage

Run the `smrs` script with the following command:

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
