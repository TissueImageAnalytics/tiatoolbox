OPENSLIDE_PATH = r"D:\\Dropbox\\PhD_Work\\PythonVE\\openslide-win64-20171122\\bin"
import os

os.add_dll_directory(OPENSLIDE_PATH)
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, filters, measure, morphology

from tiatoolbox.tools.registration.wsi_registration import (
    Registration,
    RegistrationParameters,
)
from tiatoolbox.wsicore.wsireader import WSIReader


def get_mask(grayscale):
    # grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(grayscale, np.mean(grayscale), 255, cv2.THRESH_BINARY)
    mask = morphology.remove_small_objects(mask == 0, min_size=1000, connectivity=2)
    mask = morphology.binary_opening(mask, morphology.disk(3))
    mask = morphology.remove_small_objects(mask == 1, min_size=1000, connectivity=2)
    mask = morphology.remove_small_holes(mask, area_threshold=100)

    # remove all the objects while keep the biggest object only
    label_img = measure.label(mask)
    regions = measure.regionprops(label_img)
    mask = mask.astype(bool)
    all_area = [i.area for i in regions]
    second_max = max([i for i in all_area if i != max(all_area)])
    mask = morphology.remove_small_objects(mask, min_size=second_max + 1)
    mask = mask.astype(np.uint8)
    return mask


def main():
    data_dir = "D:\\Dropbox\\PhD_Work\\PROJECTS\\Registration_Project\\MEVIS\\Case1"
    target_wsi_name = "Case1_Section1.tif"
    source_wsi_name = "Case1_Section2.tif"
    target_wsi_path = "%s/%s" % (data_dir, target_wsi_name)
    source_wsi_path = "%s/%s" % (data_dir, source_wsi_name)

    params = RegistrationParameters()
    pre_alignment, dfbr_refinement = Registration.run_registration(
        target_wsi_path, source_wsi_path, params
    )
    print(pre_alignment)
    print(dfbr_refinement)


if __name__ == "__main__":
    main()
