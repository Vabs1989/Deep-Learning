###########
# IMPORTS #
###########
import pandas as pd, os, glob
from keras.applications.densenet import DenseNet121 as arch
from keras.optimizers import Adam as optimizer
from keras.losses import categorical_crossentropy as loss_func
from keras.activations import sigmoid as activation
from keras.layers import Dense as Dense
from keras.models import Model as Model
from keras_preprocessing.image import ImageDataGenerator

##########
# PARAMS #
##########

# DIR AND FILES
metadata_dir = "/home/administrator/PycharmProjects/densenet_121/meta_data"
train_filename = "All_Training_70Patient_structured.csv"
valid_filename = "All_Validation_10Patient_structured.csv"
img_dir = "/home/administrator/Documents/NIH Dataset/resized_only/i224"

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
no_of_classes = len(selected_labels)

# IMAGE RELATED PARAMS
img_size = 224 # if include_top is False , img_size = 224
no_of_channels = 3 # if weights = imagenet, 3 channels are mandatory

# RUN RELATED PARAMS
mode = "simple"
train_batch_size = 16
valid_batch_size = 16
# mode = "weighted_avg"

to_normalize = False
to_flip = False

initial_epoch = 0
end_epoch = 200


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

# LOSS PARAMS
patience = 10


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

# THIS LEGENDARY CODE THAT GOES OUTTA MEMORY
# IMAGE LOADER
train_datagen=ImageDataGenerator()
valid_datagen=ImageDataGenerator()

train_generator=train_datagen.flow_from_dataframe(
dataframe=train_data,
# directory="./miml_dataset/images",
x_col="filepath",
y_col=selected_labels,
batch_size=train_batch_size,
seed=42,
shuffle=True,
class_mode="other",
target_size=(img_size,img_size))

valid_generator=valid_datagen.flow_from_dataframe(
dataframe=valid_data,
# directory="./miml_dataset/images",
x_col="filepath",
y_col=selected_labels,
batch_size=valid_batch_size,
seed=42,
shuffle=True,
class_mode="other",
target_size=(img_size,img_size))

# TRAIN MODEL
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
print(STEP_SIZE_TRAIN)
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
print(STEP_SIZE_VALID)
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    # validation_data=valid_generator,
                    # validation_steps=STEP_SIZE_VALID,
                    epochs=3
)
# THIS LEGENDARY CODE THAT GOES OUTTA MEMORY

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()