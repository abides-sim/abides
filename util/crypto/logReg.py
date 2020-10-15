from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score as f1
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import mean_squared_error as mse
from scipy.special import expit
import numpy as np

np.set_printoptions(linewidth=120, precision=4, threshold=np.inf)

def getWeights(previous_weight = None, max_iter = 5, lr = 1.0,  trainX = None, trainY = None, self_id = None):
    """
    getWeights initializes a logistic regresssion model with the average weights from the previous iteration,
    then trains the model using mini-batch gradient descent for max_iter iterations with learning rate lr,
    on local training data subset/partition trainX, trainY.  self_id should be the agent simulation id.
    """

    # Census dataset uses -1 for False, but our math expects 0 for False.  Fix it up.
    trainY[trainY < 0] = 0

    # If there was not a previous iteration of the protocol, initialize all weights to zero.
    # Otherwise, initialize to the average weights from the previous iteration.
    if previous_weight is None:
      weight = np.zeros(trainX.shape[1])
    else:
      weight = previous_weight.copy()

    # Train the model for max_iter iterations with learning rate lr.
    weight = np_train(weight, trainX, trainY, lr, max_iter)

    ### Uncomment next two lines for each client to print local training accuracy each iteration.  Will be slower.
    #pred = predict_all(trainX, weight)
    #print (f"Client {self_id} local acc {acc(trainY, np.array(pred)):0.3f}.")

    # Return the local weights from this client training only on its own local data.
    return weight


def reportStats(weight, current_iteration, X_train, y_train, X_test, y_test):

    y_train[y_train < 0] = 0
    y_test[y_test < 0] = 0

    ypred_is = predict_all(X_train, weight)
    ypred_oos = predict_all(X_test, weight)

    np_err_handling = np.seterr(invalid = 'ignore')

    is_acc = acc(y_train, ypred_is)
    is_mcc = mcc(y_train, ypred_is)
    is_f1 = f1(y_train, ypred_is)
    is_mse = mse(y_train, ypred_is)

    oos_acc = acc(y_test, ypred_oos)
    oos_mcc = mcc(y_test, ypred_oos)
    oos_f1 = f1(y_test, ypred_oos)
    oos_mse = mse(y_test, ypred_oos)

    is_tn, is_fp, is_fn, is_tp = confusion_matrix(y_train, ypred_is).ravel()
    oos_tn, oos_fp, oos_fn, oos_tp = confusion_matrix(y_test, ypred_oos).ravel()
    is_auprc = auprc(y_train, ypred_is)
    oos_auprc = auprc(y_test, ypred_oos)

    np.seterr(**np_err_handling)

    print (f"Consensus {current_iteration}: IS acc {is_acc:0.5f}.  IS MCC {is_mcc:0.5f}.  IS F1 {is_f1:0.5f}.  IS MSE {is_mse:0.5f}.  OOS acc {oos_acc:0.5f}.  OOS MCC {oos_mcc:0.5f}.  OOS F1 {oos_f1:0.5f}.  OOS MSE {oos_mse:0.5f}.")
    print (f"Confusion {current_iteration}: IS TP: {is_tp}, IS FP: {is_fp}, IS TN: {is_tn}, IS FN: {is_fn}, IS AUPRC: {is_auprc:0.5f}.  OOS TP: {oos_tp}, OOS FP: {oos_fp}, OOS TN: {oos_tn}, OOS FN: {oos_fn}, OOS AUPRC: {oos_auprc:0.5f}.")

    return is_acc, is_mcc, is_f1, is_mse, is_auprc, oos_acc, oos_mcc, oos_f1, oos_mse, oos_auprc


def np_predict_all(X, weight):
  w = np.tile(weight, (X.shape[0],1))
  pred = np.einsum('ij,ij->i', X, w)
  return expit(pred)


def np_train(weight, trainX, trainY, lr, n):
  for i in range(n):
    sum_error = 0
    m = np_predict_all(trainX, weight)
    e = trainY - m
    g = (e * m * (1.0 - m)).reshape(-1,1) * trainX
    ag = np.mean(g, axis=0)
    weight = weight + lr * ag
    se = np.sum(e ** 2)

    #print(f"new>epoch={i}, lr={lr:0.3f}, error={se:0.3f}")

  return weight


def predict_all(trainX, weight):
    pred_raw = np_predict_all(trainX, weight)

    pred = np.zeros(pred_raw.shape)
    pred[pred_raw > 0.5] = 1

    return pred

