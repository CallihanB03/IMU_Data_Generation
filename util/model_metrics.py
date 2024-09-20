import torch
from util.label_encoding import one_hot_to_dummy_encoding

def precision(model, input_data, labels, precison_wrt):
    """
    precision = TP / (TP + FP)
    """
    precison_wrt = int(precison_wrt)
    tp_count = 0
    fp_count = 0
    pred_vals = model(input_data)

    pred_vals = one_hot_to_dummy_encoding(pred_vals)
    labels = one_hot_to_dummy_encoding(labels)

    assert pred_vals.shape == labels.shape, f"shape of predicted labels {pred_vals.shape} is incompatible with shape of true labels {labels.shape}"
        

    for i, _ in enumerate(pred_vals):
        pred_val = int(pred_vals[i].item())
        true_val = int(labels[i].item())

        if pred_val == precison_wrt and true_val == precison_wrt:
            tp_count += 1

        elif pred_val == precison_wrt and true_val != precison_wrt:
            fp_count += 1

    return round(tp_count / (tp_count + fp_count), 2)



    



def recall(model, input_data, labels, recall_wrt):
    """
    recall = TP / (TP + FN)
    """
    recall_wrt = int(recall_wrt)

    tp_count = 0
    fn_count = 0
    pred_vals = model(input_data)

    pred_vals = one_hot_to_dummy_encoding(pred_vals)
    labels = one_hot_to_dummy_encoding(labels)

    assert pred_vals.shape == labels.shape, f"shape of predicted labels {pred_vals.shape} is incompatible with shape of true labels {labels.shape}"

    for i, _ in enumerate(pred_vals):
        pred_val = int(pred_vals[i].item())
        true_val = int(labels[i].item())

        if pred_val == recall_wrt and true_val == recall_wrt:
            tp_count += 1
        
        elif pred_val != recall_wrt and true_val == recall_wrt:
            fn_count += 1

    return round(tp_count / (tp_count + fn_count), 2)


    

def f1(model, input_data, labels, f1_wrt):
    f1_wrt = int(f1_wrt)

    prec = precision(model, input_data, labels, precison_wrt=f1_wrt)
    rec = recall(model, input_data, labels, recall_wrt=f1_wrt)

    return round(2 * ((prec * rec) / (prec + rec)), 2)

if __name__ == "__main__":
    from util.participant_data import load_participant_data
    from util.pre_processing import *
    from util.label_encoding import *
    from classification_models.simple_classification import Classifier


    data = load_participant_data()

    data = data[data.columns[:-2]] # Removing string arguments
    X, y = df_to_tensor(data[data.columns[:-1]]), df_to_tensor(data[data.columns[-1]])
    y = dummy_to_one_hot(y)


    x_train, y_train, x_test, y_test = tensor_train_test_split(X, y, test_size=0.2)

    input_dim = x_train.shape[1]
    num_classes = 3



    classifier = Classifier(input_dim=input_dim, 
                            hidden_dim=24, 
                            num_classes=num_classes)
    
    label=2
    classifier_prec = precision(classifier, x_test, y_test, precison_wrt=label)
    classifier_rec = recall(classifier, x_test, y_test, recall_wrt=label)
    classifier_f1 = f1(classifier, x_test, y_test, f1_wrt=label)

    print(f"precision = {classifier_prec}")
    print(f"recall = {classifier_rec}")
    print(f"f1 = {classifier_f1}")
    
