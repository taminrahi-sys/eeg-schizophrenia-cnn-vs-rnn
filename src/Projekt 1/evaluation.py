from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



def evaluate(y_true, y_pred, y_prob):

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm
    }