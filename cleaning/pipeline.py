#!/usr/bin/env python3
import os
import stat
import pathlib
import argparse
import json
import logging
from datetime import datetime
import subprocess
import uuid
import inspect
import sys
import tarfile
from tqdm import tqdm
from typing import Optional, Iterator, Tuple
from functools import wraps
from dataclasses import dataclass
from contextlib import contextmanager

"""
UTIL
"""

LOG_FORMAT = "[%(levelname)s] %(filename)s: %(message)s"

# track stage index
stage_index = 0

def get_stage_num() -> int:
    global stage_index
    return stage_index

# instance ID env var key
INSTANCE_ID_KEY = "SWARM_PIPELINE_INSTANCE_ID"

def stage_only(f):
    """
    mark functions that should not be called inside this file
    """
    @wraps(f)
    def wrapper(*args, **kwds):
        caller = inspect.getmodule(inspect.currentframe().f_back)
        if caller is sys.modules[__name__]:
            raise RuntimeError(f"{f.__name__} cannot be called from inside the pipeline module")
        return f(*args, **kwds)
    return wrapper

def stdout_message_prefix() -> str:
    instance_id = os.environ.get(INSTANCE_ID_KEY)
    return f"[INSTANCE-OUTPUT-{instance_id}] "

def log_stage_output(data: dict) -> None:
    print(f"\n{stdout_message_prefix()}{json.dumps(data)}")

"""
EXPORTS
"""

@dataclass
class StageContext:
    meta_path: pathlib.Path
    working_dir: pathlib.Path
    meta: Optional[dict] = None
    output_dir: Optional[pathlib.Path] = None

    def __post_init__(self):
        self.meta = json.loads(self.meta_path.read_text())

    def update(self) -> None:
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def rows(self, discard_invalid: bool = True) -> Iterator[Tuple]:
        for row in self.meta[self.working_dir.parent.stem]:
            if discard_invalid and not row["valid"]:
                continue

            yield (
                row,
                self.working_dir / row["name"], 
                self.output_dir / row["name"] if self.output_dir is not None else None,
            )

    def row_count(self, discard_invalid: bool = True) -> int:
        count = 0
        for row in self.meta[self.working_dir.parent.stem]:
            if discard_invalid and not row["valid"]:
                continue
            count += 1
        return count

    def make_output_dir(self, name: str = f"stage{get_stage_num()}") -> None:
        """
        get path for output directory if writing to one
        notify pipeline that image set was modified
        """
        # protect raw images
        assert name != "raw"
        self.output_dir = self.meta_path.parent / name
        self.output_dir.mkdir()
        log_stage_output({"mutated": True, "name": name})

@stage_only
def get_stage_context() -> StageContext:
    """
    create input arguments for pipelined scripts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meta", required=True, type=str, help="specify metadata file path")
    parser.add_argument("-w", "--working-dir", required=True, type=str, help="specify working directory")
    parser.add_argument("-l", "--log-level", required=False, type=str, default="ERROR", help="specify log level")
    args = parser.parse_args()
    
    # propagate the log level
    logging.basicConfig(
        format=LOG_FORMAT,
        level=getattr(logging, args.log_level, logging.ERROR),
    )

    return StageContext(meta_path=pathlib.Path(args.meta), working_dir=pathlib.Path(args.working_dir))

"""
MAIN
"""

# what is considered "data"
DATA_SUFFIXES = set([".png"])

# generate name to identify pipeline batch
BATCH_ID = datetime.now().strftime("b%Y%m%d_%H%M%S")

def generate_batch_name(id: str) -> str:
    return f"pipeline-meta-{id}.json"

def generate_metadata_obj(path: pathlib.Path) -> dict:
    """
    generate JSON metadata for working folder
    """
    # load root labels (e.g. if data partitioned across features already)
    root_path = path / "pipeline-root.json"
    base_labels = {}
    if root_path.exists():
        pipeline_root = root_path.read_text()
        pipeline_root = json.loads(pipeline_root)

        for feature, value in pipeline_root.get("base_labels", {}).items():
            base_labels[feature] = value

    raw_path = path / "raw"

    # data is probably compressed, try to extract it for future use
    if not raw_path.exists() or (raw_path.is_dir() and not any(raw_path.iterdir())):
        raw_path.mkdir(exist_ok=True)

        # look for gzip archive and create raw/
        archive_path = path / "raw.tar.gz"
        assert archive_path.exists()

        with tarfile.open(archive_path, "r") as tar:
            total_size = sum(mem.size for mem in tar.getmembers())

            with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Extracing {archive_path.name}") as bar:
                for mem in tar.getmembers():
                    if not mem.isfile():
                        continue        
                
                    out_path = pathlib.Path(mem.name)
                    if not out_path.suffix.lower() in DATA_SUFFIXES:
                        continue
                        
                    # "flatten" the path and skip duplicates
                    out_path = raw_path / out_path.name
                    if out_path.exists():
                        continue

                    # block traversals
                    if not out_path.resolve().is_relative_to(raw_path.resolve()):
                        continue

                    # write all files to raw/
                    with tar.extractfile(mem) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                    bar.update(mem.size)

    meta = dict([(path.name, [])])
    for f in raw_path.iterdir():
        if f.suffix.lower() in DATA_SUFFIXES:
            meta[path.name].append({
                "name": f.name,
                "valid": True,
                "labels": base_labels,
            })
    
    return meta

def write_new_files(new_files: list[str]) -> None:
    pathlib.Path("./.latest-pipeline-new-files").write_text(
        " ".join(new_files)
    )

def get_metadata(args: argparse.Namespace) -> pathlib.Path:
    """
    create new metadata file or find existing one.
    return path to metadata file
    """
    base_path = pathlib.Path(args.working_dir)
    path = base_path / generate_batch_name(args.batch_id)
        
    if args.meta:
        # create the batch if specified by CLIF
        logging.info("specified meta option, generating new metadata file...")
        
        # make directories if they don't exist
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)

        # panic if file exists already
        path.touch(exist_ok=False)
        with path.open("w", encoding="utf-8") as f:
            json.dump(generate_metadata_obj(base_path), f, indent=2)
    else:
        assert path.exists()
    
    return path

def interpret_stage_output(stdout: str) -> dict:
    """
    determine if image set was mutated or not
    """
    message_prefix = stdout_message_prefix()
    for line in stdout.splitlines():
        if line.strip().startswith(message_prefix):
            result = json.loads(line[len(message_prefix):])
            return result
    return {}

def main():
    global stage_index

    # CLI
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="pipeline cleaning scripts",
    )

    parser.add_argument("-m", "--meta", required=False, action="store_true", default=False, help="generate base metadata file for working directory")
    parser.add_argument("-w", "--working-dir", required=False, type=str, default=".", help="specify working directory")
    parser.add_argument("-b", "--batch-id", required=False, type=str, default=BATCH_ID, help="specify pipeline batch ID")
    parser.add_argument("-l", "--log-level", required=False, type=str, default="ERROR", help="specify log level")
    parser.add_argument("stages", nargs="+", help="specify pipeline stages (list of space-separated scripts)")
    args = parser.parse_args()

    logging.basicConfig(
        format=LOG_FORMAT,
        level=getattr(logging, args.log_level, logging.ERROR),
    )

    # store state info
    new_files = []

    # get metadata file
    meta_path = get_metadata(args)
    new_files.append(str(meta_path.resolve()))
    write_new_files(new_files)

    # run stages
    os.environ[INSTANCE_ID_KEY] = str(uuid.uuid4())
    working_dir = meta_path.parent / "raw"
    
    for stage in args.stages:
        # get working directory path
        assert working_dir.exists() and working_dir.is_dir()

        stage = pathlib.Path(stage)
        result = subprocess.run(
            [
                stage.resolve(), 
                "--meta", str(meta_path.resolve()), 
                "--working-dir", str(working_dir.resolve()),
                "--log-level", args.log_level,
            ],
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        # update working directory
        output = interpret_stage_output(result.stdout)
        if output.get("mutated", False):
            working_dir = working_dir.with_name(output.get("name", stage.stem))
            logging.info(f"new working directory: {str(working_dir)}")

            new_files.append(working_dir.name)
            write_new_files(new_files)

        # track stages
        stage_index += 1

if __name__ == "__main__":
    main()