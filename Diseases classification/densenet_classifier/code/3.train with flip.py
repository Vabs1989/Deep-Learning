# https://github.com/keras-team/keras/issues/6444
###########
# IMPORTS #
###########
import datetime
import pandas as pd, numpy as np
import os, glob
from keras.applications.densenet import DenseNet121 as arch
from keras.optimizers import Adam as optimizer
# from keras.losses import categorical_crossentropy as loss_func
from keras.losses import binary_crossentropy as loss_func
# from keras.losses import binary_crossentropy
from keras.activations import sigmoid as activation
from keras.layers import Dense as Dense
from keras.models import Model as Model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, Callback
from sklearn.metrics import roc_auc_score
import keras.backend as K
# import tensorflow as tf


##########
# PARAMS #
##########
print(datetime.datetime.now())
# DIR AND FILES

metadata_dir = "/home/administrator/PycharmProjects/densenet_121/meta_data"
train_filename = "All_Training_70Patient_structured.csv"
valid_filename = "All_Validation_10Patient_structured.csv"
img_dir = "/home/administrator/Documents/NIH Dataset/resized/i224"
logs_path = "/home/administrator/PycharmProjects/densenet_121/output/log.csv"
save_weights_path = "/home/administrator/PycharmProjects/densenet_121/output/weights-{epoch:05d}-{avg_of_class_aucs:.3f}.hdf5"

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
img_size = 224 # if include_top is False , img_size = 224
no_of_channels = 3 # if weights = imagenet, 3 channels are mandatory

# RUN RELATED PARAMS
mode = "simple"
# mode = "weighted_avg"
train_batch_size = 16
valid_batch_size = None

to_normalize = False
to_flip = True
flip_perc = 0.5

initial_epoch = 0
end_epoch = 300
num_train_samples = 19845
# num_train_samples = "complete_set"`````````````````````````````````````````````````````````````
num_valid_samples = "complete_set"

# MODEL RELATED PARAMS
# LAYERS AND WEIGHTS
pooling = "avg"
# pooling = "max"

# weights = "/home/administrator/PycharmProjects/densenet_121/resume_weights/weights-00050-0.182.hdf5"
weights = "imagenet"

# OPTIMIZER PARAMS
init_lr = 1e-3
beta_1=0.9
beta_2=0.999
decay=0.0
patience = 10
factor = 0.1
min_delta = 0.001

csvlogger_callback = CSVLogger(logs_path, append=True)

checkpoint_callback = ModelCheckpoint(save_weights_path,
                                      monitor='val_acc',
                                      # monitor='avg_of_class_aucs',
                                      verbose=2,
                                      save_best_only=False,
                                      mode='max')

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    # monitor='avg_of_class_aucs',
    factor=factor, patience=patience,
    verbose=1, mode='auto',
    min_delta=0.001, cooldown=0,
    min_lr=0)

def blank_function(self, logs=None):
    return

reduce_lr.on_train_begin = blank_function
########
# CODE #
########

def configure_model(arch, weights, input_shape, pooling, no_of_classes, loss_func, optimizer):
    model = arch(
        include_top=False,
        weights=weights,
        # input_tensor=None,
        input_shape=input_shape,
        pooling=pooling)

    final_layer = Dense(no_of_classes, activation=activation)
    final_layer_output = final_layer(model.output)
    model = Model(inputs=model.input, outputs=final_layer_output)

    if weights is not None and weights != "imagenet":
        model.load_weights(weights)

    model.compile(optimizer=optimizer,
                  loss=loss_func,
                  metrics=['acc'])

    return model

def get_optimizer(optimizer, init_lr, beta_1, beta_2):
    optimizer = optimizer(lr=init_lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
    return optimizer

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

def load_sample_and_labels(df, filepath_column, num_train_samples, y_colnames, used_image_ids = []):
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

# GET DATA
train_data = pd.read_csv(os.path.join(metadata_dir, train_filename))
valid_data = pd.read_csv(os.path.join(metadata_dir, valid_filename))

all_img_paths = glob.glob(os.path.join(img_dir,"**/*.npy"),recursive=True)
print(len(all_img_paths))

all_img_paths_df = pd.DataFrame(None,columns=[primary_key_column,"filepath"])
all_img_paths_df.loc[:, "filepath"] = all_img_paths
all_img_paths_df.loc[:, primary_key_column] = all_img_paths_df.loc[:, "filepath"].apply(get_filename_from_filepath)

train_data = pd.merge(train_data, all_img_paths_df, on=primary_key_column).drop(primary_key_column+".1",axis=1)
valid_data = pd.merge(valid_data, all_img_paths_df, on=primary_key_column).drop(primary_key_column+".1",axis=1)

# LABEL COUNTS
count_of_labels = pd.concat([train_data.loc[:, selected_labels].sum(),
                             valid_data.loc[:, selected_labels].sum()], axis=1)
count_of_labels.columns = ["train","valid"]
print(count_of_labels)

if train_batch_size is None:
    train_batch_size = count_of_labels["train"].sum(axis=0)

if valid_batch_size is None:
    valid_batch_size = count_of_labels["valid"].sum(axis=0)

# MODE AND WEIGHTS DICTIONARY
print("mode = ", mode)
class_weight_dict = {}
if mode == "weighted_avg":
    class_weight_dict = max(count_of_labels.loc[:, "train"]) / count_of_labels.loc[:, "train"]
    class_weight_dict = dict(class_weight_dict)
    class_weight_dict
    print(class_weight_dict)
else:
    class_weight_dict = None

# OPTIMIZER
optimizer = get_optimizer(optimizer, init_lr, beta_1, beta_2)

# MODEL
model = configure_model(arch, weights,
                        [img_size, img_size, no_of_channels],
                        pooling, no_of_classes,
                        loss_func, optimizer)

# print(model.summary())

# IMAGE LOADER
used_image_ids_valid = []
if num_valid_samples == "complete_set":
    num_valid_samples = len(valid_data.index)
X_valid, Y_valid, used_image_ids_valid = load_sample_and_labels(
    valid_data, filepath_column, num_valid_samples,
    selected_labels, used_image_ids_valid)

used_image_ids_train = []

class Custom_Predictions(Callback):
    def on_train_begin(self, logs={}):
        self.curr_preds = None
        self.classwise_aucs = []
        self.avg_of_class_aucs = []

    def on_epoch_end(self, epoch, logs):
        curr_epoch_aucs = []
        curr_trgt = self.validation_data[1]

        # y_true = K.variable(curr_trgt)

        curr_trgt = pd.DataFrame(curr_trgt)
        curr_preds = self.model.predict(self.validation_data[0])

        # y_pred = K.variable(curr_preds)
        curr_preds = pd.DataFrame(curr_preds)

        # error = K.eval(binary_crossentropy(y_true, y_pred))
        # logs["bin_cro_entpy"] = error
        # curr_auc = roc_auc_score(curr_trgt, curr_preds)
        self.curr_preds = curr_preds
        no_of_classes = curr_trgt.shape[1]
        for i in range(no_of_classes):
            test_y = curr_trgt.iloc[:, i]
            prob_y = curr_preds.iloc[:, i]
            curr_auc = roc_auc_score(test_y, prob_y)
            curr_epoch_aucs.append(curr_auc)
        avg_auc_curr_epoch = sum(curr_epoch_aucs)/no_of_classes
        curr_epoch_aucs.append(avg_auc_curr_epoch)
        print(curr_epoch_aucs)
        self.classwise_aucs.append(curr_epoch_aucs)
        self.avg_of_class_aucs.append(avg_auc_curr_epoch)
        print(" val_auc: {} ".format(avg_auc_curr_epoch))
        logs["classwise_auc"] = curr_epoch_aucs
        logs["avg_of_class_aucs"] = avg_auc_curr_epoch
        logs["lr"] = K.eval(self.model.optimizer.lr)
        answer = pd.concat([curr_trgt, curr_preds],axis=1)
        file_name = os.path.join("/home/administrator/PycharmProjects/densenet_121/output","preds_{}.csv".format(epoch))
        answer.to_csv(file_name)
        return
valid_preds = Custom_Predictions()
callbacks_list = [
    valid_preds,
    csvlogger_callback,
    checkpoint_callback,
    reduce_lr
]

#TO REMOVE IF LOOPED
if num_train_samples == "complete_set":
    num_train_samples = len(train_data.index)
#TO REMOVE IF LOOPED

# TO LOOP #
hist_df = pd.DataFrame(None)
for iter in range(initial_epoch, end_epoch):
    X_train, Y_train, used_image_ids_train = load_sample_and_labels(
        train_data, filepath_column, num_train_samples,
        selected_labels, used_image_ids_train)

    # IMAGE AUGMENTATION
    if to_normalize:
        print("THE DATA IS BEING NORMALISED")
        X_train = normalize_by_imagenet(X_train,"densenet")
        X_valid = normalize_by_imagenet(X_valid,"densenet")

    if to_flip:
        for i in range(X_train.shape[0]):
            if np.random.randint(0, high=100, size=1) < (flip_perc*100):
                print("FLIPPING..",i)
                curr_img = X_train[i, :, :, 0]
                X_train[i, :, :, 0] = np.flip(curr_img.copy(), 1)


    # TO LOOP #
    history = model.fit(X_train, Y_train,
                        batch_size=train_batch_size,
                        initial_epoch=iter,
                        epochs=iter+1,
                        # epochs=1,
                        verbose=1,
                        # shuffle=True,
                        callbacks=callbacks_list,
                        class_weight=class_weight_dict,
                        validation_data=(X_valid, Y_valid))
    # convert the history.history dict to a pandas DataFrame:
    hist_df_curr = pd.DataFrame(history.history)
    hist_df_curr.index = range(iter, iter+1)
    hist_df = pd.concat([hist_df, hist_df_curr],axis=0)

print(datetime.datetime.now())

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(hist_df['acc'])
plt.plot(hist_df['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(hist_df['loss'])
plt.plot(hist_df['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot auc & lr values
plt.plot(hist_df['avg_of_class_aucs'])
plt.plot(hist_df['lr'])
plt.title('AUC and lr')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(['AUC', 'LR'], loc='upper left')
plt.show()