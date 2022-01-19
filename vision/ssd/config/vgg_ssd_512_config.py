import numpy as np
from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 512
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(64, 8, SSDBoxSizes(51, 102), [2]),
    SSDSpec(32, 16, SSDBoxSizes(102, 188), [2, 3]),
    SSDSpec(16, 32, SSDBoxSizes(188, 274), [2, 3]),
    SSDSpec(8, 64, SSDBoxSizes(274, 360), [2, 3]),
    SSDSpec(6, 124, SSDBoxSizes(360, 446), [2]),
    SSDSpec(4, 256, SSDBoxSizes(446, 532), [2])
]

# specs = [
#     SSDSpec(64, 8, SSDBoxSizes(36, 77), [2]),
#     SSDSpec(32, 16, SSDBoxSizes(77, 154), [2, 3]),
#     SSDSpec(16, 32, SSDBoxSizes(154, 230), [2, 3]),
#     SSDSpec(8, 64, SSDBoxSizes(230, 307), [2, 3]),
#     SSDSpec(4, 128, SSDBoxSizes(307, 384), [2, 3]),
#     SSDSpec(2, 256, SSDBoxSizes(384, 460), [2]),
#     SSDSpec(1, 512, SSDBoxSizes(460, 537), [2])
# ]



priors = generate_ssd_priors(specs, image_size)