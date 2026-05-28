import cv2
from vision.geography_detector import GeographyDetector

img = cv2.imread('logs/debug/debug_000801.png')
geo = GeographyDetector()
geo.update(img)

# Save mask for manual verification
cv2.imwrite('logs/debug/river_mask.png', geo.river_mask)

# Test a line crossing the river (bottom left to top right)
crosses = geo.crosses_river((100, 900), (1800, 200))
print(f"Test crossing: {crosses}")
