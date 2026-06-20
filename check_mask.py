import cv2
import numpy as np

mask = cv2.imread(
    r"LandCoverAI/output/masks/M-33-20-D-c-4-2_0.png",
    cv2.IMREAD_UNCHANGED
)

print("Shape:", mask.shape)
print("Dtype:", mask.dtype)
print("Unique Values:")
print(np.unique(mask))