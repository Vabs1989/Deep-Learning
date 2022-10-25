###########
# IMPORTS #
###########
import pandas as pd
from sklearn.preprocessing import label_binarize

##########
# PARAMS #
##########

train_filename = "/home/administrator/PycharmProjects/densenet_121/meta_data/All_Training_70Patient.csv"
valid_filename = "/home/administrator/PycharmProjects/densenet_121/meta_data/All_Validation_10Patient.csv"
test_filename = "/home/administrator/PycharmProjects/densenet_121/meta_data/All_Testing_20Patient.csv"

label_column = "label_name"
primary_key_column = "patientId"

########
# CODE #
########

def binarize_labels(df,label_names):
    binary_label_df = label_binarize(df.loc[:, label_column],label_names)
    binary_label_df= pd.DataFrame(binary_label_df,columns=label_names)
    return binary_label_df

def append_binarized_labels_to_metadata(metadata,binary_labels):
    metadata = pd.concat([metadata, binary_labels], axis=1)
    groupby_obj = metadata.groupby(primary_key_column)
    multi_label_train = groupby_obj[label_names].sum()

    unique_meta_data_train = groupby_obj[meta_data_column_names].first()
    metadata = pd.concat([unique_meta_data_train, multi_label_train], axis=1)
    return metadata

# READ DATA
train_metadata = pd.read_csv(train_filename)
valid_metadata = pd.read_csv(valid_filename)
test_metadata = pd.read_csv(test_filename)

# GET LABEL NAMES AND META DATA COLUMN NAMES
label_names = train_metadata.loc[:, label_column].unique()
meta_data_column_names = list(set(train_metadata.columns) - set(label_column) - set(label_names) - set(primary_key_column))

# BINARIZE LABELS
binary_label_df_train = binarize_labels(train_metadata, label_names)
binary_label_df_valid = binarize_labels(valid_metadata, label_names)
binary_label_df_test = binarize_labels(test_metadata, label_names)

# PRINT THE DATASET LABEL COUNTS
count_of_labels = pd.concat([binary_label_df_train.sum(),
                             binary_label_df_valid.sum(),
                             binary_label_df_test.sum()], axis=1)
count_of_labels.columns = ["train","valid","test"]

print("DATASET has following COUNTS of each LABEL :\n", count_of_labels,"\n\n")
print("TOTAL by FILE : \n",count_of_labels.sum(axis=0))
print("GRAND TOTAL : ",count_of_labels.sum(axis=0).sum(),"\n\n\n")


# JOIN THE APPROPRIATE METADATA WITH THE BINARIZED LABELS
train_metadata = append_binarized_labels_to_metadata(train_metadata,binary_label_df_train)
valid_metadata = append_binarized_labels_to_metadata(valid_metadata,binary_label_df_valid)
test_metadata = append_binarized_labels_to_metadata(test_metadata,binary_label_df_test)

print("UNIQUE records by file :\n")
print("Train : ", len(train_metadata.index))
print("Valid : ", len(valid_metadata.index))
print("Test : ", len(test_metadata.index))
print("Total unique records : ",
      sum([
          len(train_metadata.index),
          len(valid_metadata.index),
          len(test_metadata.index)
      ]),
      "\n\n\n")

# DROP ORIGINAL SINGLE LABEL COLUMN, AS IT HAS BEEN BINARIZED
train_metadata.drop(label_column,axis = 1,inplace = True)
valid_metadata.drop(label_column,axis = 1,inplace = True)
test_metadata.drop(label_column,axis = 1,inplace = True)

train_metadata.to_csv(train_filename.replace(".csv", "_structured.csv"))
valid_metadata.to_csv(valid_filename.replace(".csv", "_structured.csv"))
test_metadata.to_csv(test_filename.replace(".csv", "_structured.csv"))
print("Data Structuring Complete....")



