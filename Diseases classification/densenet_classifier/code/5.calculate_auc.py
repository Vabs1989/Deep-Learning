import pandas as pd, os
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
predictions_path = '/home/administrator/PycharmProjects/densenet_121/output'
answer = pd.DataFrame()
# initial_epoch = 0
# end_epoch = 3

filenames = [
    "flipping_test_preds_e00011.csv",
    "flipping_train_preds_e00011.csv",
    "flipping_valid_preds_e00011.csv"
]
# for iter_no in range(initial_epoch,end_epoch):
    # file_path = os.path.join(predictions_path, 'predictions_e{}.csv'.format(str(iter_no).zfill(5)))
for iter_no in filenames:
    file_path = os.path.join(predictions_path, iter_no)
    print(file_path)
    valid_true_pred_df = pd.read_csv(file_path)
    no_of_classes = int(len(valid_true_pred_df.columns)/2)
    curr_epoch_aucs = []
    for i in range(no_of_classes):
        test_y = valid_true_pred_df.iloc[:, i]
        prob_y = valid_true_pred_df.iloc[:, no_of_classes + i]
        auc = roc_auc_score(test_y, prob_y)
        curr_epoch_aucs.append(auc)
    average_auc = sum(curr_epoch_aucs)/no_of_classes
    curr_epoch_aucs.append(average_auc)
    answer = pd.concat([answer,pd.DataFrame(curr_epoch_aucs).transpose()],axis=0)
answer.to_csv(os.path.join(predictions_path,'..',"all_aucs.csv"))