#!/usr/bin/env python3
import cv2
import logging
import pipeline

# accepts folder name "raw", for each valid image, process it & edit metadata.
def main():
    ctx = pipeline.get_stage_context()
    
    # track greens
    bad_images_count = 0
    total_count = 0

    # =====PROCESSING=====
    lower = (0, 154, 0) 
    upper = (0, 255, 0) 
    for data, src, _ in ctx.rows():
        image = cv2.imread(src)

        mask = cv2.inRange(image, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        # count how many non-zero pixels are in the mask
        count = cv2.countNonZero(mask)

        # If there's a significant amount of pure green pixels, we can safley assume it was corrupted
        # Set label to be false
        if count > 500:
            bad_images_count += 1
            data["valid"] = False
            data["labels"]["is_green"] =  True    
        
        total_count += 1
    # =====FINISHED=====

    logging.info(f"{bad_images_count} out of {total_count} images had green error pixels: {round(bad_images_count/total_count*100, 2)}% error rate")
    
    # update metadata
    ctx.update()
    
if __name__ == "__main__":
    main()