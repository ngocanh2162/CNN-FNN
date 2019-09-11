import numpy as np
from scipy import sparse
import pickle
from nltk import metrics
from sklearn import metrics

def initFilter(N1, N2): 
    W = []
    b = []
    n = len(N1)
    for i in range(n):
        W.append(0.01*np.random.rand(N1[i], N2)) # N[i], X.shape[1]
        b.append(0.01*np.random.rand(1,1))
    return (W,b) #len(Nfilter)*[k[i],X.shape[2]]
  
def initW(N1, N2):
    W = 0.01*np.random.rand(N1, N2)# N[i], X.shape[1]
    b = 0.01*np.random.rand(N1,1)
    return (W,b)

def convert_labels(y, C): #ma trận với mỗi cột ô ở hàng = correctclass có giá trị=1 các hàng còn lại =0
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

def reLU(A): 
    return np.maximum(0,A)

def leakyReLU(A, alpha = 0.01):
    dx = np.ones_like(A)
    dx[A < 0] = alpha
    return dx*A

def softmax(V): 
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

def conv_forward(A, K, b): #K:len(filter)*[filter[i], X.shape[2]]
    m, H_old, W_old = A.shape #m: record
    C = []
    for i in range(len(K)):
        f1, f2 = K[i].shape
        H_new = H_old - f1 + 1 
        W_new = W_old - f2 + 1 
        A_res = np.zeros((m, H_new, W_new))
        for j in range(m):
            for h in range(H_new):
                for v in range(W_new):
                    a_slide = A[j, h: h + f1, v:v+f2]
                    A_res[j,h,v] = np.sum(a_slide * K[i][:,:]) + b[i][:,:]
        C.append(A_res)
    return C 

def maxPooling1d(A): #len(filter)*[m, filter'[i], 1]
    m = A[0].shape[0]
    temp1 = np.zeros((m, len(A), 1))
    temp2 = np.zeros((m, len(A), 1), dtype = int)
    for i in range(m):
        for j in range(len(A)):
            temp1[i,j] = np.amax(A[j][i,:])
            temp2[i,j] = np.argmax(A[j][i,:])
    return (temp1,temp2)

def CNN(X, Y, Nloop, N1, lr = 1): # X[records, words, features]
    m = X.shape[0]
    K, b = initFilter(N1, X.shape[2]) #len(filter)*[filter[i], X.shape[2]]
    W, b0 = initW(Y.shape[1], len(Nfilter)) 
    for i in range(Nloop):
        print(i,1)
        C = conv_forward(X, K, b) #len(filter)*[m, filter'[i], 1]
        Cre = list()
        for j in range(len(C)):
           Cre.append(leakyReLU(C[j]))
        f, indexF = maxPooling1d(Cre)       #[m, len(filter), 1]
        dW = np.zeros_like(W)
        db0 = np.zeros_like(b0)
        dC = []
        dK = []
        db = []
        M = list()
        Yhat = np.zeros((m, 2, 1))
        for j in range(m):
            M.append(np.dot(W,f[j]) + b0)   #[2, 1]
            U = np.argmax(softmax(M[-1]))
            Yhat[j] = convert_labels([U], 2)  #[m, 2, 1]
        print(i,2)
        E = (Yhat -Y)/X.shape[2] #[m, 2, 1]
        for j in range(m):
            dW += E[j].dot(f[j].T) 
            db0 += np.sum(E[j], axis=1, keepdims = True)
        W += -lr*dW
        b0 += -lr*db0
        print(i,3)
        for k in range(len(K)):
            temp1 = C[k].shape[1] #[m, filter'[i], 1]
            temp2 = K[k].shape[0] #[filter[i], X.shape[2]]
            dK = np.zeros_like(K[k]) #[filter[i], X.shape[2]]
            db = np.zeros_like(b[k])
            dC = np.zeros_like(C[k]) #[m, filter'[i], 1]
            for j in range(m):
                dC[j,indexF[j,k]] = 1 
                hstart = 0
                X[j] = np.rot90(X[j],2)
                for h in range(temp1):
                    x_slide = X[j, h: h + temp2, :]
                    dK += x_slide * dC[j,h,:] 
                    db += dC[j,h,:]
            K[k] += -lr*dK
            b[k] += -lr*db
    return(K, b, W, b0)
    
def pre(X, K, b, W, b0):
    m = X.shape[0]
    C = conv_forward(X, K, b) 
    Cre = list()
    for j in range(len(C)):
         Cre.append(leakyReLU(C[j])) 
    f, indexF = maxPooling1d(Cre) 
    Yhat = list()
    for j in range(m):
        M = np.dot(W,f[j]) + b0   #[m, 2, 1]
        Yhat.append(np.argmax(softmax(M))) #[m,1]
    print(Yhat)
    return Yhat

x_train = pickle.load(open('Trained Data/imdb/x_train.pkl', 'rb'))
y_train = pickle.load(open('Trained Data/imdb/y_train.pkl', 'rb'))
x_test = pickle.load(open('Trained Data/imdb/x_test.pkl', 'rb'))
y_test = pickle.load(open('Trained Data/imdb/y_test.pkl', 'rb'))

Nloop = 5
Nfilter = [8,5,8,5]
Y = np.zeros((len(y_train), 2, 1))
for i in range(len(y_train)):
    Y[i] = convert_labels([y_train[i]], 2)
K, b, W, b0 = CNN(x_train, Y, Nloop, Nfilter, 0.1)
preY = pre(x_test, K, b, W, b0)
print('Training accuracy: %.2f %%' % (100*metrics.accuracy_score(preY, y_test)))