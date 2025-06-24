# Sparse Modeling Representative Selection
Generating Sufficient Querysets for Information Pursuit using Sparse Representative Modeling

This project was created using uv. 
To prepare the environment after cloning, run:
uv sync

Run smrs with this script:
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