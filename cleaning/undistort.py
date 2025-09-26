#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import pipeline
import logging
import pathlib

def main():
    ctx = pipeline.get_stage_context()
    ctx.make_output_dir("undistorted")

    # load camera calibration
    calib = pathlib.Path("../collections/armor_plates_9-14-25/calibrations")
    assert calib.exists()

    calib = next(calib.iterdir())
    logging.info(f"loading calibration: {str(calib)}")
    calib = yaml.load(calib.read_text(), Loader=yaml.Loader)

    w, h = calib["Resolution"]
    K = np.array(calib["K"])
    D = np.array(calib["D"])

    x_map, y_map = cv2.fisheye.initUndistortRectifyMap(K, D, R=np.eye(3), P=K.copy(), size=(w, h), m1type=cv2.CV_16SC2)
    
    # iterate over data rows
    for data, src, dst in ctx.rows():
        image = cv2.imread(src)
        fixed = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR)
        data["labels"]["undistorted"] = True
        cv2.imwrite(dst, fixed)

    # update metadata
    ctx.update()

if __name__ == "__main__":
    main()