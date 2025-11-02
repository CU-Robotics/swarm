# Swarm Machine-Learning Toolkit

Swarm packages the data-cleaning pipeline, annotation helpers, and training scripts that CU Robotics uses to build perception models for Hive. The repository is organized around four main areas: cleaning raw collections, preparing datasets, training/evaluating models, and running exploratory notebooks.

## Prerequisites
- Python 3.11+ with `venv`.
- System packages required by OpenCV’s GUI backends (for example, `sudo apt install libgl1` on Ubuntu).
- A PyTorch build appropriate for your CPU/GPU (install separately via <https://pytorch.org/get-started/locally/>).

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# install PyTorch separately for your hardware, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Working with Collections
Collections live under `collections/<name>/` and typically contain:
- `calibrations/` — camera YAML files used by undistortion stages.
- `examples/` — `raw.tar.gz` plus an optional `pipeline-root.json` that seeds metadata.
- Pipeline outputs produced by the scripts below.

Run the default cleaning pipeline (destroys and rebuilds transient outputs each time):
```bash
cd cleaning
./default_pipeline.sh /abs/path/to/collection/examples -l INFO
```
You can execute individual stages with `./pipeline.py` (see inline help for options) and reset outputs using `python destroy_pipeline_state.py`. Stage implementations live in `cleaning/*.py`; the Flask bounding-box annotator is served by `label_bbox.py`.

## Building Datasets and Training
```bash
cd cnn
python combine_metadata.py ../collections/<dataset>
python clean_metadata.py ../collections/<dataset>
python train_model.py ../collections/<dataset>
python test_model.py ../collections/<dataset>
```
- `combine_metadata.py` aggregates metadata from cleaned batches.
- `clean_metadata.py` discards rows that reference missing files.
- `train_model.py` instantiates the ConvNet defined in `ConvNet.py` and saves checkpoints.
- `test_model.py` evaluates the saved weights and reports accuracy/misclassifications.

`run_model.py` offers a real-time inference demo against a connected camera (`/dev/stereo-cam-right-video` by default). Adjust device paths or color thresholds as needed.

## Notebooks
Exploratory analyses live under `jupyter/` (for example, `inspect_plates.ipynb` for cropped plates, `undistort.ipynb` for calibration experiments). Activate the virtual environment before launching Jupyter.

## Contribution Checklist
- Keep pipeline contracts stable—coordinate schema or tooling changes with the ML tooling maintainers before merging.
- Document new stages or scripts directly in code docstrings or a short note in `cleaning/README.md`.
- Capture representative metrics or plots (training curves, confusion matrices, etc.) when submitting model or pipeline changes.
- Run relevant scripts end-to-end (cleaning + training) and include the commands plus key results in your PR description.
