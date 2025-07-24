import torch
import numpy as np
from PIL import Image
from skimage.morphology import binary_dilation, disk
from romiseg.segmentation_2d import fileset_segmentation
from plantdb.commons.test_database import test_database