"""
Gaze Correction Network
Task: Predict 2D gaze correction residuals from pixel coordinates
Input: 
    default is: avg original gaze point (x, y) pixel coordinates, range ~[0, 2000]*[0, 1000] 
    or a set of collected gaze points (x, y) pixel coordinates, range ~[0, 2000]*[0, 1000]
Output: (dx, dy) residual corrections, range ~[-50, 50]*[-50, 50]

Key Design Principles:
1. Simple MLP architecture
2. Input normalization for numerical stability (pixel -> normalized range)
3. Output scaling to match residual magnitude
4. Residual connections for better gradient flow
"""




