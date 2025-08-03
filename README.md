# **Generating Sufficient Querysets for Information Pursuit using Sparse Representative Modeling**

This repository contains the Python implementation (using PyTorch) of the Sparse Modeling Representative Selection (SMRS) algorithm proposed by Elhamifar et al. in [See All by Looking at A Few](http://www.vision.jhu.edu/assets/ElhamifarCVPR12.pdf).
It was developed as part of the Bachelor's Thesis of **Carl Kemmerich**, supervised by Prof. Dr. Gitta Kutyniok and Stefan Kolek M.Sc. at the Bavarian AI Chair for Mathematical Foundations of Artificial Intelligence, Ludwig-Maximilians-Universität Munich.

---
## Project Overview
This project uses SMRS to automatically select informative and diverse queries for use in explainable models like Information Pursuit (IP). It leverages CLIP embeddings to compute cosine similarities between images and natural language queries.

The goal is to reduce the size of the queryset without sacrificing informativeness, which is especially useful for large-scale vision tasks where annotation or query design is expensive.

---
## Research Context & Results


This implementation:
- Enables sparse selection of queries from a larger candidate set.
- Improves computational efficiency and interpretability in IP.
- Was validated using CIFAR-10, where SMRS successfully selected representative queries that outperform random subsets in informativeness.

More details can be found in the thesis PDF, available upon request.

---

## Contributions

The repository contains the following contributions:
- PyTorch implementation of the SMRS algorithm (based on Elhamifar et al.).
- Integration with CLIP for computing image-query cosine similarities.
- Scripts for:
    - Running SMRS on a custom query set and (a subset of) CIFAR-10 images
    - Running SMRS on standard datasets 
- Instructions for generating input query sets

--- 

## Getting Started

This project uses `uv` for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/carlk13/smrs.git
    cd smrs
    ```
2.  **Install dependencies:**

    If you don't have `uv` installed, run:
    ```bash
    pip install uv
    ```
    Then install all dependencies and set up the environment with:
    ```bash
    uv sync
    ```

---

## Usage
### Run smrs on a cosine similarity matrix (Queryset × CIFAR-10)
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
- `100`= number of queries
- `10` = images per class

### Run smrs on a standard dataset
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
```

### Generating a Query Set 
Use this prompt to generate a semantically meaningful queryset tailored for IP on CIFAR-10
```
Information Pursuit (IP) is an explainable prediction algorithm that greedily selects a sequence of interpretable queries about the data in order of information gain, updating its posterior at each step based on observed query-answer pairs. The algorithm requires a queryset of task-relevant queries. For example if the class labels are
- car 
- horse 
- boat 
- bird
then a good queryset could contain queries such as
- four-legs 
- water 
- wheels 
- wings 
- metal body 
- living thing 
- animal 
- street 
- traffic light 
Note the queries should probe the presence of semantically meaningful things but never ask exactly for the class label, i.e. the query should not be a synonym of the class label car. Next I will give you the class labels and I ask you to output exactly 2000 queries. Please output them in a way so that I can immediately save them as a .txt, i.e. each query is in a new line and they are not enumerated.
 ------------------------ Class labels ------------------------
Airplanes
Cars
Birds
Cats
Deer
Dogs
Frogs
Horses
Ships
Trucks

```
    

