import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OneHotEncoder

def check_fitted(clf): 
    return hasattr(clf, "classes_")


def create_roc_curve_plot(clf, X_train, X_test, y_train, y_test, figname=None, legend="ROC Curve"):


    if False and len(y_train.shape) == 1:
        # enc = OneHotEncoder(handle_unknown='ignore')
        # enc.fit(np.concatenate([y_train, y_test]))

        # y_train = enc.transform([y_train).toarray()
        # y_test = enc.transform([y_test).toarray()

        classes_list = list(set(np.concatenate([y_train, y_test])))
        # print(set(np.concatenate([y_train, y_test])))
        # print(set(y_train))
        # print(set(y_test))
        # print(classes_list)
        y_train = label_binarize(y_train, classes=classes_list)
        y_test = label_binarize(y_test, classes=classes_list)

    # print(y_train.shape)
    # print(y_test.shape)
    # assert y_train.shape[1] == y_test.shape[1]
    # y_score = clf.fit(X_train, y_train).decision_function(X_test)

    if check_fitted(clf):
        # print("Is fitted")
        y_score = clf.predict_proba(X_test)
    else:
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    y_score = np.array(y_score)
    # print(y_score.shape)


    # n_classes = y_train.shape[1]
    n_classes = len(set(np.concatenate([y_train, y_test])))

    
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # y_test_bin = np.zeros((len(y_test), 2))
    # new_y_score = np.zeros((len(y_test), 2))
    
    y_test_bin = np.zeros((len(y_test), n_classes))
    new_y_score = np.zeros((len(y_test), n_classes))

    idx_normal = np.where(np.array(clf.classes_)=='normal')[0][0]
    new_y_score[:, 0] = y_score[:, idx_normal]
    new_y_score[:, 1] = y_score[:, 1-idx_normal]
    y_score = new_y_score
    
    y_test_bin[y_test == 'normal', 0] = 1
    # y_test_bin[y_test != 'normal', 1] = 1
    list_classes = list(set(np.concatenate([y_train, y_test]))-{'normal'})
    list_classes = sorted(list_classes)

    for i, c in enumerate(list_classes):
        y_test_bin[y_test == c, i+1] = 1




    # print(y_test)
    # print(y_test_bin[:5])
    # print(y_score[:5])
    
    for i in range(n_classes):
        # fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    if n_classes > 2:
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # # Then interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(n_classes):
        #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # # Finally average it and compute AUC
        # mean_tpr /= n_classes

        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        return roc_auc["micro"]


    

    fig, ax = plt.subplots()
    lw = 2
    ax.plot(
        fpr[0],
        tpr[0],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[0],
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(legend)
    ax.legend(loc="lower right")

    # print(figname)

    if figname is None:
        plt.show()
    else:
        fig.savefig(figname)

    # return roc_auc[0]
    return roc_auc["micro"]

def main():
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    # y = label_binarize(y, classes=[0, 1, 2])
    # n_classes = y.shape[1]
    n_classes = len(set(y))
    
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=random_state)
    )

    create_roc_curve_plot(classifier, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()