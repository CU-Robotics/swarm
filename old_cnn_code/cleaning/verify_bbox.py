#!/usr/bin/env python3
import cv2
import sys
import numpy as np
import logging
import pipeline
from util import TextBox, silence_stderr, KeyCode
from tqdm import tqdm

BASE_BORDER = 1
SELECT_BORDER = BASE_BORDER + 1

# review image bounding boxes and discard invalid ones
def main():
    ctx = pipeline.get_stage_context()
    rows = list(ctx.rows())
    row_count = ctx.row_count()

    edits = []
    for data, _, _ in rows:
        edits.append([True] * len(data["labels"].get("plates", [])))

    if row_count == 0:
        sys.exit(1)

    done = False
    image_index = 0

    with tqdm(total=ctx.row_count(), unit="img", desc=f"Correcting auto-detected plates") as bar:
        while not done:
            data, src, _ = rows[image_index]
            undist = cv2.imread(str(src))

            plates = data["labels"].get("plates", [])
            assert len(plates) > 0, "zero detections should not pass through filter"

            plate_index = 0
            first_view = True
            while True:
                image = undist.copy()
    
                for i, (x, y, w, h) in enumerate(plates):
                    # bold borders if viewing image for first time
                    if first_view:
                        selected = edits[image_index][i]
                    else:
                        selected = i == plate_index

                    border = SELECT_BORDER if selected else BASE_BORDER
                    cscale = 1 if selected else 0.5
                    color = (0, 255*cscale, 0) if edits[image_index][i] else (0, 0, 255*cscale)
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, border)

                rsz = cv2.resize(image, None, fx=0.5, fy=0.5)
                
                with TextBox(rsz) as tb:
                    tb.write("press '<-' or '->' to select box; 't' to toggle keep; 'sp' to skip; 'esc' to finish")
                    tb.write(f"{data["name"]}", end=" ")
                    tb.write(f"{image_index + 1}/{row_count}", color=(0, 255, 255))
                    tb.write("keep: ", end="")
                    for i, keep in enumerate(edits[image_index]):
                        selected = i == plate_index
                        icon = "T" if keep else "F"
                        color = (0, 255, 0) if keep else (0, 0, 255)

                        if selected:
                            icon = f"[{icon}]"

                        tb.write(icon, color=color, end="")

                with silence_stderr():
                    cv2.imshow("review image", rsz)

                # wait on keypress
                key = cv2.waitKey(0) & 0xff
                if key == KeyCode.LEFT_ARROW or key == ord('a'):
                    if not first_view:
                        plate_index = (plate_index - 1) % len(plates)
                elif key == KeyCode.RIGHT_ARROW or key == ord('d'):
                    if not first_view:
                        plate_index = (plate_index + 1) % len(plates)
                elif key == ord("t"):
                    # write edits; toggle validity of plate detection
                    keep = edits[image_index][plate_index]
                    edits[image_index][plate_index] = not keep
                elif key == ord(" "):
                    # queue next image
                    image_index = (image_index + 1) % row_count
                    bar.n = max(bar.n, image_index + 1)
                    bar.refresh()
                    break
                elif key == KeyCode.BACKSPACE:
                    image_index = (image_index - 1) % row_count
                    bar.n = max(bar.n, image_index + 1)
                    bar.refresh()
                    break
                elif key == KeyCode.ESCAPE:
                    bar.n = ctx.row_count()
                    bar.refresh()
                    done = True
                    break

                first_view = False

    # commit changes
    for edit, (data, _, _) in zip(edits, rows):
        plates = data["labels"].get("plates", [])
        plates = [plate for plate, keep in zip(plates, edit) if keep]
        data["labels"]["plates"] = plates
        if len(plates) == 0:
            data["valid"] = False

    # update metadata
    ctx.update()

if __name__ == "__main__":
    main()