# Swarm Machine-Learning Toolkit

End-to-end tooling for turning raw Hive match footage into deployable armor-plate classifiers. The repository packages image-cleaning pipelines, labeling utilities, training scripts, sample datasets, and reference models so teams can spin up a complete workflow quickly.

## Highlights
- Stage-based cleaning pipeline that can resume, branch, and replay on any collection.
- Browser-based bounding-box annotator and OpenCV QA tools for rapid human review.
- Torch model training, evaluation, and deployment scripts tied to the cleaned dataset format.
- Sample collections, camera calibrations, and pretrained weights to bootstrap experimentation.

## Repository Layout
- `cleaning/` – pipeline driver, reusable stages, and helper scripts for managing image collections.
- `cnn/` – dataset utilities, ConvNet definition, and CLI scripts for training/evaluating models.
- `collections/` – example collections (raw frames, metadata seeds, calibrations) that mirror the expected on-disk layout.
- `jupyter/` – exploratory notebooks for inspecting frames, calibrations, and distortion fixes.
- `models/` – stored PyTorch checkpoints (`full_model.pth`, `best_model.pth`) used by `cnn/run_model.py`.
- `requirements.txt` – Python dependencies for the pipeline and orchestration scripts (install PyTorch separately for your platform).

## Getting Started

### Prerequisites
- Python 3.11+ recommended.
- System packages for OpenCV GUI backends (`sudo apt install libgl1` on Ubuntu).
- [PyTorch](https://pytorch.org/get-started/locally/) installed with the accelerator that matches your hardware (not pinned in `requirements.txt`).

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# install PyTorch separately, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Working With Data Collections

Each collection lives in a dedicated folder (for example `collections/armor_plates_9-14-25/`) with the structure:

```
collection/
├── calibrations/              # camera calibration YAMLs used by undistortion
├── examples/
│   ├── pipeline-root.json     # base labels: color, icon, etc.
│   └── raw.tar.gz             # archived source frames (extracted on first run)
└── (pipeline outputs)
```

Before running the pipeline on your own dataset:
1. Copy the collection folder to a writable location (or mount a new directory with the same layout).
2. Place `raw.tar.gz` alongside an optional `pipeline-root.json` that seeds base labels (`icon`, `color`, and arbitrary metadata).
3. Confirm `cleaning/config.yaml` matches the lighting and color thresholds for the match you captured.

## Cleaning Pipeline

The orchestration script `cleaning/pipeline.py` chains executable “stages.” Each stage receives a shared metadata JSON file and the current working directory containing the active image set (`StageContext` in `cleaning/pipeline.py:60`). Stages opt into writing mutated outputs via `ctx.make_output_dir(...)`, which swaps the downstream working directory and records new folders in `.latest-pipeline-new-files`.

### Run the Default Pipeline
```bash
cd cleaning
./default_pipeline.sh /absolute/path/to/collection/examples -l INFO
```
The helper script resets previously generated folders (`cleaning/default_pipeline.sh:20`) and executes the recommended sequence:

1. `kill_green.py` – drops frames with green corruption artifacts (`cleaning/kill_green.py:7`).
2. `undistort.py` – applies fisheye undistortion using the latest calibration (`cleaning/undistort.py:10`).
3. `plate_detector.py` – auto-detects armor plates via HSV masking heuristics (`cleaning/plate_detector.py:40`).
4. `label_bbox.py` – serves a Flask-based bounding-box annotator at `http://localhost:8000` (`cleaning/label_bbox.py:21`).
5. `crop.py` – crops, squares, and grayscales each accepted plate (`cleaning/crop.py:11`).

You can run any subset manually:
```bash
./pipeline.py -w /path/to/collection/examples run_stage.py another_stage.py
```
Use `-m` to mint a fresh metadata file (`pipeline-meta-<timestamp>.json`) and `-b` to reuse a prior batch.

### Manual Labeling Workflow
- Visit `http://localhost:8000` once `label_bbox.py` starts the annotator to draw, delete, and save bounding boxes. Press “Save” periodically to persist edits (`cleaning/label_bbox.py:66`).
- To review auto-generated boxes with keyboard controls (toggle keep, skip, revisit), run `./pipeline.py ... verify_bbox.py` after detection (`cleaning/verify_bbox.py:14`).

### Metadata Outputs
- Metadata lives in `pipeline-meta-<batch>.json` under the collection root. Each row stores `name`, `valid`, and `labels`, which includes per-frame annotations and provenance like `undistort_calib_batch` (`cleaning/pipeline.py:188`).
- Cropped assets reside in the stage-named directory (default `cropped/`) beside the metadata file.
- To wipe transient outputs and redo the pipeline, call `cleaning/destroy_pipeline_state.py` (protects the original `raw/` directory by design, `cleaning/destroy_pipeline_state.py:18`).

## Building a Training Dataset

Once you have one or more cleaned batches:
```bash
cd cnn
python combine_metadata.py ../collections/your_dataset
python clean_metadata.py ../collections/your_dataset
```
- `combine_metadata.py` merges every valid detection, flattening multi-plate frames into separate entries while preserving folder provenance (`cnn/combine_metadata.py:18`).
- `clean_metadata.py` drops metadata rows that no longer have a cropped image on disk (`cnn/clean_metadata.py:29`).

The resulting `cleaned_metadata.json` and the per-folder `cropped/` directories are what the dataset loader expects.

## Training and Evaluating the Model

Run supervised training on the cleaned dataset:
```bash
python train_model.py ../collections/your_dataset
```
- The script instantiates `cnn/ConvNet.py` (`cnn/ConvNet.py:4`) and trains with 80/20 splits, logging progress with `tqdm` (`cnn/train_model.py:57`).
- A PyTorch state dictionary is saved to `cifar_net.pth` by default (`cnn/train_model.py:125`).

To evaluate (and optionally visualize mistakes):
```bash
python test_model.py ../collections/your_dataset
```
`cnn/test_model.py` reloads the saved weights, computes accuracy, and plots misclassified samples using Matplotlib (`cnn/test_model.py:78`).

## Real-Time Inference Demo

`cnn/run_model.py` streams frames from `/dev/stereo-cam-right-video`, reuses the HSV detector, crops, and classifies armor plates using pretrained weights (`cnn/run_model.py:123`). Adjust the `detect_armor_plates` color parameter or camera device path to match your setup. The script currently writes a preview frame to `cnn/TEST.png` for debugging (`cnn/run_model.py:149`).

## Jupyter Notebooks
- `jupyter/inspect_plates.ipynb` – quick visualization of cropped assets for sanity checks.
- `jupyter/undistort.ipynb` – experiments with camera intrinsics and fisheye rectification.

Launch with your preferred environment (`jupyter lab` or `jupyter notebook`) after activating the virtual environment.

## Extending the Pipeline
- Follow the contract exposed by `pipeline.get_stage_context()` to access rows and update metadata (`cleaning/pipeline.py:106`).
- Use `ctx.rows(filter_by={True, False})` to iterate over invalid frames as well (`cleaning/pipeline.py:74`).
- Call `ctx.make_output_dir("my-stage")` before writing transformed assets so downstream stages use your outputs automatically (`cleaning/pipeline.py:95`).
- Store any stage-specific configuration in `cleaning/config.yaml` or a sibling YAML to keep the CLI flags simple.

## License

This project is released under the MIT License. See `LICENSE` for details.
