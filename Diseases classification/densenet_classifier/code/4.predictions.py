###########
# IMPORTS #
###########
import datetime
import pandas as pd, numpy as np
import os, glob
from keras.applications.densenet import DenseNet121 as arch
from keras.optimizers import Adam as optimizer
from keras.activations import sigmoid as activation
from keras.layers import Dense as Dense
from keras.models import Model as Model

##########
# PARAMS #
##########

# DIR AND FILES
metadata_dir = "/home/administrator/PycharmProjects/densenet_121/meta_data"
img_dir = "/home/administrator/Documents/NIH Dataset/resized/i224"

test_filename = "All_Training_70Patient_structured.csv"
suffix = "flipping_train"


# LABELS AND CLASSES
selected_labels = [
    # "No_Finding",
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax'
]
label_column = "label_name"
primary_key_column = "patientId"
filepath_column = "filepath"
no_of_classes = len(selected_labels)

# IMAGE RELATED PARAMS
img_size = 224  # if include_top is False , img_size = 224
no_of_channels = 3  # if weights = imagenet, 3 channels are mandatory

# RUN RELATED PARAMS
to_normalize = False
to_flip = False

# MODEL RELATED PARAMS
# LAYERS AND WEIGHTS
pooling = "avg"
# pooling = "max"

weights = "/home/administrator/PycharmProjects/densenet_121/prediction_weights/weights-00011-0.785.hdf5"

########
# CODE #
########

def get_filename_from_filepath(some_path):
    file_name = some_path.split(os.path.sep)[-1].split(".")[0]
    return file_name


def convert_to_three_layer(single_channel_image_array, axis=2):
    # axis = the axis on which we have /supposed to have the channels [zero indexed]
    # for 1 image with channels last pass axis = 2
    # for n images with channels last pass axis = 3

    if single_channel_image_array.ndim == axis:
        # If image does not have the dimension for single layer
        single_channel_image_array = np.expand_dims(single_channel_image_array, axis=-1)
    # three_channel_image_array = np.concatenate(
    #     (single_channel_image_array, single_channel_image_array, single_channel_image_array), axis=axis)
    three_channel_image_array = np.repeat(single_channel_image_array, 3, axis=axis)
    return three_channel_image_array


def load_sample_and_labels(df, filepath_column, num_train_samples, y_colnames, used_image_ids=[]):
    train_images = df.loc[:, filepath_column]
    if len(train_images) < num_train_samples:
        train_sample_images = np.random.choice(train_images, num_train_samples, replace=True)
        train_sample_array = np.asarray([np.load(arr) for arr in train_sample_images])
        train_sample_array = convert_to_three_layer(train_sample_array, 3)
        return train_sample_array, []

    # Sets the {(train images) minus (images used in previous iterations)} as the current sample
    # Also use of set causes it to take unique images only
    train_sample_images = list(set(train_images) - set(used_image_ids))

    if len(train_sample_images) < num_train_samples:
        sample_diff = num_train_samples - len(train_sample_images)
        # set all the images, which are not in current sample (previously used or unused) as not_sampled
        not_sampled = list(set(train_images) - set(train_sample_images))
        train_sample_images.extend(np.random.choice(not_sampled, sample_diff, replace=False))
        # since you exhausted all your images , reset the used set of images
        used_image_ids = []
    else:
        train_sample_images = np.random.choice(train_sample_images, num_train_samples, replace=False)

    used_image_ids.extend(train_sample_images)
    train_sample_ids = [_.split("/")[-1].split(".")[0] for _ in train_sample_images]

    train_sample_df = df[(df.patientId.isin(train_sample_ids))]
    train_sample_df.index = train_sample_df.patientId

    train_sample_labels = np.asarray(train_sample_df[y_colnames])
    train_sample_images = train_sample_df.loc[:, filepath_column]
    train_sample_array = np.asarray([np.load(arr) for arr in train_sample_images])
    train_sample_array = convert_to_three_layer(train_sample_array, 3)
    return train_sample_array, train_sample_labels, used_image_ids


def normalize_by_imagenet(x, model):
    x = x.astype("float16")
    if x.ndim == 3:
        if model in ("inception", "xception", "mobilenet"):
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
        elif model in ("resnet", "vgg"):
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
        elif model in ("resnet", "vgg"):
            x -= 115.799
    return x


# GET DATA
test_data = pd.read_csv(os.path.join(metadata_dir, test_filename))

all_img_paths = glob.glob(os.path.join(img_dir, "**/*.npy"), recursive=True)
print(len(all_img_paths))

all_img_paths_df = pd.DataFrame(None, columns=[primary_key_column, "filepath"])
all_img_paths_df.loc[:, "filepath"] = all_img_paths
all_img_paths_df.loc[:, primary_key_column] = all_img_paths_df.loc[:, "filepath"].apply(get_filename_from_filepath)

test_data = pd.merge(test_data, all_img_paths_df, on=primary_key_column).drop(primary_key_column + ".1", axis=1)

# LABEL COUNTS
count_of_labels = pd.concat([test_data.loc[:, selected_labels].sum()], axis=1)
count_of_labels.columns = ["train"]
print(count_of_labels)

model = arch(
    include_top=False,
    weights="imagenet",
    input_shape=[img_size, img_size, no_of_channels],
    pooling=pooling)

final_layer = Dense(no_of_classes, activation=activation)
final_layer_output = final_layer(model.output)
model = Model(inputs=model.input, outputs=final_layer_output)

model.load_weights(weights)

# IMAGE LOADER
used_image_ids_valid = []
num_test_samples = len(test_data.index)
X_test, Y_test, used_image_ids_valid = load_sample_and_labels(
    test_data, filepath_column, num_test_samples,
    selected_labels, used_image_ids_valid)
Y_test = pd.DataFrame(Y_test)
# IMAGE AUGMENTATION
if to_normalize:
    print("THE DATA IS BEING NORMALISED")
    X_test = normalize_by_imagenet(X_test, "densenet")

if to_flip:
    for i in range(X_test.shape[0]):
        if np.random.binomial(1, 0.5):
            curr_img = X_test[i, :, :, 0]
            X_test[i, :, :, 0] = np.flip(curr_img.copy(), 1)

pred_probs = model.predict(X_test)
pred_probs = pd.DataFrame(pred_probs)

result = pd.concat([Y_test, pred_probs], axis=1, ignore_index=True)
result.to_csv("/home/administrator/PycharmProjects/densenet_121/output/"+suffix+"_preds_e{}.csv".format(
    weights.split("-")[1]
), index=False)