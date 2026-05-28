import cv2
import numpy as np
from typing import List, Tuple

class LineEndDetector:
    """Detects the valid endpoints (T-caps) of a metro line."""

    def get_endpoints(self, line_mask: np.ndarray, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extracts the geometric endpoints from a line's skeleton.
        An endpoint in a 1-pixel thick skeleton is a pixel with exactly 1 neighbor.
        
        Args:
            line_mask: Binary mask of the colored line (for fallback checks)
            skeleton: 1-pixel thick skeleton of the line mask
            
        Returns:
            List of (x, y) coordinates for the endpoints.
        """
        # A simple convolution filter to count neighbors
        # A pixel with exactly 1 neighbor in a 3x3 neighborhood will sum to 2
        # (1 for the pixel itself + 1 for its neighbor)
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=np.uint8)
                           
        # Skeleton is 0 and 255. Divide by 255 to get 0 and 1.
        binary_skel = (skeleton > 0).astype(np.uint8)
        
        neighbor_count = cv2.filter2D(binary_skel, -1, kernel)
        
        # Endpoints are pixels in the skeleton that have exactly a count of 2
        endpoints_mask = np.logical_and(binary_skel == 1, neighbor_count == 2)
        
        ys, xs = np.where(endpoints_mask)
        
        endpoints = []
        for x, y in zip(xs, ys):
            endpoints.append((int(x), int(y)))
            
        # Sometimes small artifacts in the skeleton create fake endpoints.
        # If we have more than 2, we should probably cluster them or find the two
        # that are furthest apart, or trace the longest path in the skeleton.
        if len(endpoints) > 2:
            # Simple heuristic: find the two endpoints that are furthest apart
            max_dist = 0
            best_pair = None
            for i in range(len(endpoints)):
                for j in range(i+1, len(endpoints)):
                    dist = (endpoints[i][0] - endpoints[j][0])**2 + (endpoints[i][1] - endpoints[j][1])**2
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (endpoints[i], endpoints[j])
            
            if best_pair:
                endpoints = list(best_pair)
                
        return endpoints
