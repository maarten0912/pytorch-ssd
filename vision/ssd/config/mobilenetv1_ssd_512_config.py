import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 512 #300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# https://forums.developer.nvidia.com/t/how-train-jetson-inference-ssd512-model/168510
# https://github.com/qfgaohao/pytorch-ssd/issues/128#issuecomment-688197358
# https://github.com/dusty-nv/pytorch-ssd/commit/4ac1cfc18ac9856d247ed6a8f16a7a7de05d5364

specs = [
    SSDSpec(32, 16, SSDBoxSizes(20, 35), [2, 3]),
    SSDSpec(16, 32, SSDBoxSizes(35, 50), [2, 3]),
    SSDSpec(8, 64, SSDBoxSizes(50, 65), [2, 3]),
    SSDSpec(4, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 512), [2, 3])
]

# Generated with `python estimate_default_boxes_spec.py --net_type mb2-ssd-lite --image_size 512 --min_ratio 2 --max_ratio 50`
# specs = [
#     SSDSpec(32, 16, SSDBoxSizes(5.12, 10.24), [2, 3]),
#     SSDSpec(16, 32, SSDBoxSizes(10.24, 71.68), [2, 3]),
#     SSDSpec(8, 64, SSDBoxSizes(71.68, 133.12), [2, 3]),
#     SSDSpec(4, 128, SSDBoxSizes(133.12, 194.56), [2, 3]),
#     SSDSpec(2, 256, SSDBoxSizes(194.56, 256.0), [2, 3]),
#     SSDSpec(1, 512, SSDBoxSizes(256.0, 317.44), [2, 3])
# ]


priors = generate_ssd_priors(specs, image_size)
