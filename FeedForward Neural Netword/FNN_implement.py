import numpy as np
from scipy import sparse
import pickle
from nltk import metrics
from sklearn import metrics

def initPara(Nnode): #khởi tạo bộ W, b ở mỗi layer
    W = []
    b = []
    for i in range(len(Nnode)-1):
        W.append(0.01*np.random.randn(Nnode[i], Nnode[i+1]))
        b.append(np.zeros((Nnode[i+1], 1)))
    return (W, b)

def convert_labels(y, C): #ma trận với mỗi cột ô ở hàng = correctclass có giá trị=1 cá hàng còn lại =0
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

def activation(Y): #reLU
    return np.maximum(Y, 0)

def outputActivation(V): #softmax
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

def FFdropout(X, y, Nnode, num, drop_rate, lr = 1):
    W, b = initPara(Nnode)
    Y = convert_labels(y, Nnode[-1])
    for j in range(num):
        dW = []
        db = []
        Z = []
        O = []
        E = []
        O.append(X)

        # VD2: input = spare matrix
        Z.append(W[0].T*O[0] + b[0])
        O.append(activation(Z[-1]))
        for i in range(1, len(Nnode)-1):
        # # VD1: input = np.array
        # for i in range(len(Nnode)-1):
            Z.append(np.dot(W[i].T, O[i]) + b[i])
            m = np.random.binomial([np.ones_like(Z[-1])], 1-drop_rate)[0]
            O.append(activation(Z[-1])*m*1.0/(1-drop_rate)) # dropout
        Yhat = outputActivation(Z[-1])
        if j%100 == 0:
           print(-np.sum(Y*np.log(Yhat))/Y.shape[1]) #loss
        E.append((Yhat -Y)/X.shape[1])
        for i in range(len(Nnode)-1):
        # k = len(Nnode)-i-2 = len(W)-i-1 do đảo ngược lại dW[k], db[k], E[k] tương ứng với W[i], b[i], O[i]
            if i != 0:
                E[-1][Z[len(Nnode)-i-2] <= 0] = 0 #gradient of ReLU
            dW.append(np.array(O[len(Nnode)-i-2].dot(E[i].T)))
            db.append(np.sum( E[i], axis = 1, keepdims = True))
            E.append(np.dot(W[len(W)-i-1], E[i]))
        for i in range(len(W)):
            W[i] += -lr*dW[len(W)-i-1]
            b[i] += -lr*db[len(W)-i-1]
    return W,b

def predict(W, b, X):
    O = []
    O.append(X)
    # VD2 input = sparse matrix
    Z = W[0].T*O[0] + b[0]
    O.append(activation(Z))
    for i in range(1, len(W)):
    # # VD1 input = np.array
    # for i in range(len(W)):
        Z = np.dot(W[i].T, O[i]) + b[i]
        O.append(activation(Z))
    return np.argmax(Z, axis=0)

# #VD1
# Nnode = [2,100,50,3] #number of node in each layer include input
# N = 100 #datapoint for each class
# X = np.zeros((Nnode[0], N*Nnode[-1])) # data matrix (each row = single example)
# y = np.zeros(N*Nnode[-1], dtype='uint8') # class labels
# for j in range(Nnode[-1]):
#   ix = range(N*j,N*(j+1))
#   r = np.linspace(0.0,1,N) # radius
#   t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
#   X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
#   y[ix] = j
# W, b = FFdropout(X, y, Nnode, 1000, 0)
# print('Training accuracy: %.2f %%' % (100*np.mean(predict(W, b, X) == y)))

# #VD2:
X = pickle.load(open('Trained Data/13 classify data/vectorTrain.pkl', 'rb'))
y = pickle.load(open('Trained Data/13 classify data/y_train.pkl', 'rb'))
X_test = pickle.load(open('Trained Data/13 classify data/vectorTest.pkl', 'rb'))
y_test = pickle.load(open('Trained Data/13 classify data/y_test.pkl', 'rb'))
X = X.T
X_test = X_test.T
Nnode = [X.shape[0], 1000, 13]
W, b = FFdropout(X, y, Nnode, 1000, 0.2)
print('Training accuracy: %.2f %%' % (100*np.mean(predict(W, b, X_test) == y_test)))
