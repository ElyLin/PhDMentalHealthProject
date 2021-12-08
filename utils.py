import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



def evaluate(model, X_tr, y_tr, X_te, y_te):
    '''
    Evaluate the classification model
    
    Inputs: 
        model - model object
        X_tr - array like
            indepent variable values for training dataset
        y_tr - array like
            actual target value for training dataset
        X_te - array like
            indepent variable values for testing dataset
        y_te - array like
            actual target values for testing dataset
    
    Outputs:
        confusion matrix for testing dataset
    '''
    print('Accuracy Score:')
    print(f'Train - {accuracy_score(y_tr, model.predict(X_tr))}')
    print(f'Test - {accuracy_score(y_te, model.predict(X_te))}')
    print('  ')
    print('Precision Score:')
    print(f'Train - {precision_score(y_tr, model.predict(X_tr))}')
    print(f'Test - {precision_score(y_te, model.predict(X_te))}')
    print('  ')
    print('Recall Score: ')
    print(f'Train - {recall_score(y_tr, model.predict(X_tr))}')
    print(f'Test - {recall_score(y_te, model.predict(X_te))}')
    print('  ')
    print('F1 Score: ')
    print(f'Train - {f1_score(y_tr, model.predict(X_tr))}')
    print(f'Test - {f1_score(y_te, model.predict(X_te))}')
    print('  ')
    print('ROC_AUC: ')
    print(f'Train - {roc_auc_score(y_tr, model.predict(X_tr))}')
    print(f'Test - {roc_auc_score(y_te, model.predict(X_te))}')
    print('  ')
    print('  ')
    print('Confusion matrix for test data')
    return plot_confusion_matrix(model, X_te, y_te, normalize = 'true', include_values=True, display_labels = ["Doesn't Need Help","Need Help"], cmap=plt.cm.Blues)