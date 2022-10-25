import os, numpy as np, glob
from scipy.misc import imread
from scipy.ndimage.interpolation import zoom, rotate
from scipy.ndimage.filters import gaussian_filter


img_size = 224

# in_dir = os.path.join("/home/administrator/Documents/NIH Dataset/data", "**/images/*.png")
in_dir = os.path.join("/home/administrator/Desktop/sample (1)_format", "*.png")
# out_dir = os.path.join("/home/administrator/Documents/NIH Dataset", "resized/i{}/".format(img_size))
out_dir = in_dir + "_resized_only_i" + str(img_size)
to_preprocess = False
model = "densenet"

##########
# Resize #
##########
def resize_image(img, size, smooth=None):
    """
    Resizes image to new_length x new_length and pads with black.
    Only works with grayscale right now.

    Arguments:
      - smooth (float/None) : sigma value for Gaussian smoothing
    """
    resize_factor = float(size) / np.max(img.shape)
    if resize_factor > 1:
        # Cubic spline interpolation
        resized_img = zoom(img, resize_factor)
    else:
        # Linear interpolation
        resized_img = zoom(img, resize_factor, order=1, prefilter=False)
    if smooth is not None:
        resized_img = gaussian_filter(resized_img, sigma=smooth)
    l = resized_img.shape[0]
    w = resized_img.shape[1]
    if l != w:
        ldiff = (size - l) / 2
        wdiff = (size - w) / 2
        pad_list = [(ldiff, size - l - ldiff), (wdiff, size - w - wdiff)]
        resized_img = np.pad(resized_img, pad_list, "constant",
                             constant_values=0)
    assert size == resized_img.shape[0] == resized_img.shape[1]
    return resized_img.astype("uint8")

def preprocess_input(x, model):
    x = x.astype("float16")
    if x.ndim == 3:
        if model in ("inception","xception","mobilenet"):
            x /= 255.
            x -= 0.5
            x *= 2.
        if model in ("densenet"):
            x /= 255.
            if x.shape[-1] == 3:
                x[..., 0] -= 0.485
                x[..., 1] -= 0.456
                x[..., 2] -= 0.406
                x[..., 0] /= 0.229
                x[..., 1] /= 0.224
                x[..., 2] /= 0.225
            elif x.shape[-1] == 1:
                x[..., 0] -= 0.449
                x[..., 0] /= 0.226
        elif model in ("resnet","vgg"):
            if x.shape[-1] == 3:
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.680
            elif x.shape[-1] == 1:
                x[..., 0] -= 115.799
    if x.ndim == 2:
        x /= 255.
        if model in ("densenet"):
            x -= 0.449
            x /= 0.226
        elif model in ("resnet","vgg"):
            x -= 115.799
    return x

def resize_images_and_save_as_nparray(list_of_images, in_dir, out_dir, new_size="throw error", to_preprocess = False, model = None):
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    num_images = len(list_of_images)
    for index, img in enumerate(list_of_images):
        # sys.stdout.write("Resizing {}/{} ...\r".format(index + 1, num_images))
        # sys.stdout.flush()

        print("Resizing {}/{} ...".format(index + 1, num_images))
        loaded_img = imread(os.path.join(in_dir, img), mode="L")
        resized_img = resize_image(loaded_img, new_size)
        if to_preprocess:
            resized_img = preprocess_input(resized_img, model)

        img = img.split(os.path.sep)[-1]
        np.save(os.path.join(out_dir, img.replace("png", "npy")), resized_img)


list_of_images = glob.glob(in_dir)
resize_images_and_save_as_nparray(list_of_images, in_dir, out_dir, img_size, to_preprocess = to_preprocess, model = model)