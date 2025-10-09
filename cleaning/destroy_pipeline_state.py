#!/usr/bin/env python3
import pathlib
import shutil

# destroy new files created by most recent pipeline automatically
def main():
    state = pathlib.Path(".latest-pipeline-new-files")
    state = state.read_text().split()

    meta_path, *rest = state
    meta_path = pathlib.Path(meta_path)
    collection_path = meta_path.parent

    # delete existing meta
    # print(f"deleting metadata file: {meta_path.resolve()}")
    # meta_path.unlink(missing_ok=True)

    for dirname in rest:
        assert dirname != "raw" # protect original data
        path = collection_path / dirname
        print(f"deleting intermediate directory: {path.resolve()}")
        shutil.rmtree(path, ignore_errors=True)
    
if __name__ == "__main__":
    main()