import cv2
import numpy as np

class GeographyDetector:
    """Detects geographical features like rivers."""

    def __init__(self):
        self.river_mask = None

    def update(self, frame: np.ndarray):
        """
        Detects the river mask from the current frame.
        Handles different map styles (day/night, different cities) dynamically.
        """
        h, w = frame.shape[:2]
        
        # Exclude HUD and top margin
        top_margin = int(h * 0.06)
        hud_cutoff = int(h * 0.88)
        map_region = frame[top_margin:hud_cutoff, :]

        # 1. Smooth to reduce noise
        blurred = cv2.GaussianBlur(map_region, (11, 11), 0)

        # 2. Reshape and find unique colors
        pixels = blurred.reshape(-1, 3)
        
        # Simple color quantization to group similar colors
        pixels_quantized = (pixels // 5) * 5
        colors, counts = np.unique(pixels_quantized, axis=0, return_counts=True)

        # Sort by frequency
        sorted_idx = np.argsort(-counts)
        colors = colors[sorted_idx]
        counts = counts[sorted_idx]

        # 3. Identify the background and river colors
        # Background is the most frequent color.
        # River is usually the second most frequent contiguous color block.
        bg_color = colors[0]
        river_color = None
        
        total_pixels = len(pixels)
        for i in range(1, min(10, len(colors))):
            ratio = counts[i] / total_pixels
            # River usually occupies 3% to 20% of the screen
            if 0.03 < ratio < 0.25:
                # Check color distance to avoid picking up shadows/vignettes
                dist = np.linalg.norm(colors[i].astype(float) - bg_color.astype(float))
                if dist > 15: # Must be distinctly different from background
                    river_color = colors[i]
                    break
        
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        if river_color is not None:
            # Create a mask for the river color
            lower = np.clip(river_color.astype(int) - 10, 0, 255).astype(np.uint8)
            upper = np.clip(river_color.astype(int) + 10, 0, 255).astype(np.uint8)
            
            mask = cv2.inRange(blurred, lower, upper)
            
            # Clean up mask (remove small noise, connect blobs)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Pad mask back to full frame size
            full_mask[top_margin:hud_cutoff, :] = mask

        self.river_mask = full_mask

    def crosses_river(self, pt1: tuple, pt2: tuple) -> bool:
        """
        Checks if a line segment between pt1 and pt2 crosses the river.
        """
        if self.river_mask is None:
            return False

        # Create an empty mask to draw the line
        h, w = self.river_mask.shape
        line_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw the proposed line
        cv2.line(line_mask, pt1, pt2, 255, thickness=2)

        # Check intersection
        intersection = cv2.bitwise_and(self.river_mask, line_mask)
        return np.any(intersection > 0)
