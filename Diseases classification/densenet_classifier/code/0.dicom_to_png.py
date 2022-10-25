import numpy as np
import scipy.misc
import pydicom
import glob
import sys
import os, json

dicom_dir = "/home/administrator/Desktop/sample (1)"
to_png_dir = dicom_dir + "_format"

list_of_train_dicoms = glob.glob(os.path.join(dicom_dir,"*"))
if not os.path.exists(to_png_dir): os.makedirs(to_png_dir)

for i, each_train_dicom in enumerate(list_of_train_dicoms):
    sys.stdout.write("TRAIN DICOM to PNG conversion: {}/{} ...\r".format(i + 1, len(list_of_train_dicoms)))
    sys.stdout.flush()
    tmp_dicom_array = pydicom.read_file(each_train_dicom).pixel_array
    assert np.min(tmp_dicom_array) >= 0 & np.max(tmp_dicom_array) <= 255
    scipy.misc.imsave(os.path.join(to_png_dir, each_train_dicom.split("/")[-1].replace("dcm", "png")),
                      tmp_dicom_array)

