import sys
import os
import cv2
import numpy as np
from enum import IntEnum
from typing import Tuple
from contextlib import contextmanager

SILENCE_STDERR = True

# OpenCV and matplotlib like to dump garbage to stderr, silence it
@contextmanager
def silence_stderr():
    stderr = sys.stderr.fileno()
    devnull = os.open(os.devnull, os.O_WRONLY)

    stderr_copy = os.dup(stderr)

    if SILENCE_STDERR:
        os.dup2(devnull, stderr)

    try:
        yield
    finally:
        if SILENCE_STDERR:
            os.dup2(stderr_copy, stderr)
    
        os.close(stderr_copy)
        os.close(devnull)

class TextBox():
    def __init__(
        self,
        frame: np.ndarray,
        font: int = cv2.FONT_HERSHEY_PLAIN,
        scale: float = 1,
        thickness: int = 1,
        margin: int = 10,
        line_spacing: int = 8,
    ):
        self.frame = frame
        self.font = font
        self.scale = scale
        self.thickness = thickness
        self.margin = margin
        self.line_spacing = line_spacing
        self.cursor_x = margin
        self.cursor_y = margin
    
    def write(self, text: str, color: Tuple[int, int, int] = (255, 255, 255), end: str = "\n"):
        text += end 

        for i, text in enumerate(text.split("\n")):
            (text_width, text_height), _ = cv2.getTextSize(text, self.font, self.scale, self.thickness + 2)

            # newline
            if i > 0:
                self.cursor_y += text_height + self.line_spacing
                self.cursor_x = self.margin

            if len(text) == 0:
                continue 

            # draw text
            cv2.putText(self.frame, text, org=(self.cursor_x, self.cursor_y + text_height), fontFace=self.font, fontScale=self.scale, color=(0, 0, 0), thickness=self.thickness+2, lineType=cv2.LINE_AA)
            cv2.putText(self.frame, text, org=(self.cursor_x, self.cursor_y + text_height), fontFace=self.font, fontScale=self.scale, color=color, thickness=self.thickness, lineType=cv2.LINE_AA)

            self.cursor_x += text_width

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class KeyCode(IntEnum):
    ESCAPE = 27
    BACKSPACE = 8
    LEFT_ARROW = 81
    RIGHT_ARROW = 83