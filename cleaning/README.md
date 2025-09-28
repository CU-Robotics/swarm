# Cleaning
The data cleaning and labeling utility scripts reside here.
All utility scripts (or stages) that interact with data in some way (transform or label) must 
integrate with `pipeline.py`, which provides glue code for chaining pipeline stages.

Stages take in common input:
* Working directory: directory of images currently being processed
* Metadata path: path to metadata file that stores labels and pipeline state for a particular run of `pipeline.py`
* Logging level: all stages should use `logging` to log information. Calling `pipeline.py` propagates the global log level to all stages.

The CLI arguments are taken care of via the `get_stage_context` function in `pipeline.py`. 
The pipeline script is itself called with a list of stages to execute, in order. Stages are just executable scripts that conform to the standard above.
Stages can transform data by specifying a new output directory using its pipeline context object:
```python
ctx = pipeline.get_stage_context()
ctx.make_output_dir("name-of-output-directory")
```
It is expected that a stage that calls `make_output_dir` actually writes data to this folder, 
since all subsequent stages will use this new folder as their working directory,
unless a future stage transforms the data again. And so on.

The pipeline script allows you to dynamically exclude particular stages, reorder stages, save intermediate data in working directories, 
run partial pipelines, and resume existing pipelines from where you left off previously:
`./pipeline.py [-h] [-m] [-w WORKING_DIR] [-b BATCH_ID] [-l LOG_LEVEL] stages [stages ...]`
* `-m`: generate new metadata file (JSON).
* `WORKING_DIR` is a relative path to a project directory. To work properly, the project directory must be organized properly:
  ```
  project_root
  |__ raw/
      |__ (all source images contained here)
  |__ pipeline-root.json
  ```
  Where `project_root` must be specified as the working directory.
* `BATCH_ID` specifies the name of either a new batch (`-m`) or existing batch to use to store data: `pipeline-meta-<BATCH_ID>.json`
* `LOG_LEVEL` specifies the logging level to propagate to all stages. Use `logging` log levels.
* `stages` is a list of space-separated executable script names that conform to the stage standard above. They will be run in order,
  using previous stages' results and sharing a common metadata file to consolidate all labels.
