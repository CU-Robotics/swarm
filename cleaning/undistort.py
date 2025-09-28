#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import pipeline
import logging
import pathlib
from tqdm import tqdm

def main():
    ctx = pipeline.get_stage_context()
    ctx.make_output_dir("undistorted")

    # load camera calibration
    calib_path = pathlib.Path("../collections/armor_plates_9-14-25/calibrations")
    assert calib_path.exists()

    calib_path = next(calib_path.iterdir())
    logging.info(f"loading calibration: {str(calib_path)}")
    calib = yaml.load(calib_path.read_text(), Loader=yaml.Loader)

    w, h = calib["Resolution"]
    K = np.array(calib["K"])
    D = np.array(calib["D"])

    x_map, y_map = cv2.fisheye.initUndistortRectifyMap(K, D, R=np.eye(3), P=K.copy(), size=(w, h), m1type=cv2.CV_16SC2)
    
    # iterate over data rows
    with tqdm(total=ctx.row_count(), unit="img", desc=f"Undistorting images") as bar:
        for data, src, dst in ctx.rows():
            image = cv2.imread(src)
            fixed = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR)
            data["labels"]["undistort_calib_batch"] = calib_path.stem
            cv2.imwrite(dst, fixed)
            bar.update(1)

    # update metadata
    ctx.update()

if __name__ == "__main__":
    main()