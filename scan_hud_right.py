import cv2
import numpy as np
import os

img = cv2.imread('logs/debug/debug_000801.png')
h, w = img.shape[:2]
hud_top = int(h * 0.88)
hud_region = img[hud_top:, :]

# Let's save a visual grid of the HUD region to see what is where
grid = hud_region.copy()
for x in range(0, w, 50):
    cv2.line(grid, (x, 0), (x, grid.shape[0]), (0, 255, 0), 1)
    if x % 100 == 0:
        cv2.putText(grid, str(x), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

cv2.imwrite('logs/debug/hud_grid.png', grid)
print("Saved logs/debug/hud_grid.png with x-coordinate labels.")
