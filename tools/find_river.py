import cv2
import numpy as np

img = cv2.imread('logs/debug/debug_000801.png')

# Exclude HUD and top margin
h, w = img.shape[:2]
map_region = img[int(h*0.06):int(h*0.88), :]

# Get unique colors and counts
pixels = map_region.reshape(-1, 3)
colors, counts = np.unique(pixels, axis=0, return_counts=True)

# Sort by count (descending)
sorted_idx = np.argsort(-counts)
colors = colors[sorted_idx]
counts = counts[sorted_idx]

print("Top 10 colors in map region:")
for i in range(min(10, len(colors))):
    print(f"Color: {colors[i]}, Count: {counts[i]}, Ratio: {counts[i]/len(pixels):.3f}")

# The river is likely the 2nd most common color, or a large connected component.
# Let's save a mask of the top 3 colors to see what they are.
for i in range(min(3, len(colors))):
    mask = cv2.inRange(img, colors[i], colors[i])
    cv2.imwrite(f'logs/debug/color_{i}.png', mask)
