from sklearn.datasets import load_svmlight_file
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm 


parser = argparse.ArgumentParser(description='Gradient Free Method')
parser.add_argument('--path', default='./LIBSVM', type=str, help='path of data')
parser.add_argument('--data_file', default='mushrooms', type=str, help='file of data')
parser.add_argument('--lambda2', default=1e-5, type=float, help='coefficient of regularization')
parser.add_argument('--alpha', default=2.0, type=float, help='coefficient in the regularization')
parser.add_argument('--delta', default=0.001, type=int, help='stepsize of zeroth order gradient')
parser.add_argument('--lr', default=0.01, type=float, help='constant learning rate')
parser.add_argument('--q', default=1, type=int, help='mini batch size for zeroth order gradient in SPIDER')
parser.add_argument('--Q', default=10, type=int, help='mega batch size for zeroth order gradient in SPIDER')
parser.add_argument('--M', default=10, type=int, help='epoch length for SPIDER')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs (one pass of data)')
parser.add_argument('--print_freq', default=10, type=int, help='frequency to print train stats')
parser.add_argument('--out_fig', default= './log/', type=str, help = 'path of output picture')
    
def load_data(path, file):
    source = path + '/' + file + '.txt'
    data = load_svmlight_file(source)
    x_raw = data[0].todense()
    y = np.array(data[1])
    y = y[:,np.newaxis]

    # for the dataset that y = {1,2}
    if file in ['mushrooms', 'covtype', 'skin' ] :
        y = 2 * y -3
    
    # for datastes that y = {2,4}
    if file in ['breast']:
        y = y-3
    
    # for datasets that y = {0,1}
    if file in ['liver', 'phishing', 'svmguide']:
        y = 2* y -1 
    # for datasets that are unnormalized

    if file in ['colon', 'rna', 'medalon', 'skin', 'svmguide']:
        x_raw /= np.max(np.abs(x_raw),axis=0)

    x = np.ones([x_raw.shape[0], x_raw.shape[1] + 1])
    x[:, :-1] = x_raw
    return x, y

def cost(data, label, X, lambda2, alpha):
    n,_ = data.shape
    return np.sum( np.maximum(1 - np.matmul(data,X) * label,0) ,axis = 0) / n + lambda2 * np.sum(np.minimum(np.abs(X),alpha ), axis=0)  / n

def randSphere(d,q):
    u = np.random.randn(d,q)
    return u  / np.linalg.norm(u,axis=0)

def SGD(data,label,x,args):
    queryLst = []
    costLst = []
    n,d = data.shape
    queryCnt = 0
    elapse_time = 0.0
    for epoch in tqdm(range(args.epochs * n)):
        begin_time = time.time()
        u = randSphere(d, args.q)
        S = np.minimum(args.q,n)
        idx = np.random.choice(n,S,replace=False)
        A = data[idx,:]
        b = label[idx]
        g = d  / args.delta * u * (cost(A, b, x + args.delta * u , args.lambda2, args.alpha) - cost(A, b, x - args.delta * u , args.lambda2, args.alpha)) /2
        x = x - args.lr * np.mean(g, axis=1, keepdims=True)
        queryCnt += 1
        end_time = time.time()
        elapse_time += end_time - begin_time
        if epoch % args.print_freq == 0:
            c = cost(data, label, x, args.lambda2, args.alpha)
            print('SGD:{t:.4f},{ep:d},{loss:.8f}'.format(t=float(elapse_time), ep=epoch + 1, loss=float(c)))
            queryLst.append(queryCnt)
            costLst.append(c)
    return queryLst, costLst

def SPIDER(data,label,x,args):

    queryLst = []
    costLst = []
    n,d = data.shape
    queryCnt = 0
    elapse_time = 0.0

    for epoch in tqdm(range(args.epochs * n // (2*args.M*args.q + args.Q))):
        begin_time = time.time()
        u = randSphere(d, args.Q)
        S = np.minimum(args.Q,n)
        idx = np.random.choice(n,S,replace=False)
        A = data[idx,:]
        b = label[idx]
        v = d  / args.delta * u * (cost(A, b, x + args.delta * u , args.lambda2, args.alpha) - cost(A, b, x - args.delta * u , args.lambda2, args.alpha)) / 2
        g = np.mean(v, axis=1, keepdims=True) 
        x = x - args.lr * g
        w = x # record tyhe previous point 
        queryCnt += args.Q

        for _ in range(args.M):
            u = randSphere(d,args.q)
            S = np.minimum(args.Q,n)
            idx = np.random.choice(n,S,replace=False)
            A = data[idx,:]
            b = label[idx]
            v_current =  d  / args.delta * u * (cost(A, b, x + args.delta * u , args.lambda2, args.alpha) - cost(A, b, x - args.delta * u , args.lambda2, args.alpha)) / 2
            v_previous = d  / args.delta * u * (cost(A, b, w + args.delta * u , args.lambda2, args.alpha) - cost(A, b, w - args.delta * u , args.lambda2, args.alpha)) / 2
            g = np.mean(v_current - v_previous, axis=1, keepdims=True) + g
            x = x - args.lr * g
            w = x
            queryCnt += 2 * args.q # variance reduction query function value twice
        
        end_time = time.time()
        elapse_time += end_time - begin_time
        if epoch % args.print_freq == 0:
            c = cost(data, label, x, args.lambda2, args.alpha)
            print('SPIDER:{t:.4f},{ep:d},{loss:.8f}'.format(t=float(elapse_time), ep=epoch + 1, loss=float(c)))
            queryLst.append(queryCnt)
            costLst.append(c)
    return queryLst, costLst

def main():
    np.random.seed(42)
    args = parser.parse_args()
    data, label = load_data(args.path, args.data_file)
    _, d = data.shape
    x0 = np.ones((d,1))
    
    querySGD, costSGD = SGD(data,label,x0,args)
    querySPIDER, costSPIDER = SPIDER(data,label,x0,args)

    plt.rc('font', size=15)
    plt.figure()
    plt.semilogy(querySGD, costSGD, ':r', label='GFM', linewidth=3)
    plt.semilogy(querySPIDER, costSPIDER, '-k', label='GFM+', markerfacecolor='none', linewidth=3)
    plt.xlim((0, querySPIDER[-1]))
    plt.tick_params(labelsize=15)
    plt.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    plt.legend(fontsize=25, frameon=False)
    plt.savefig(args.out_fig + args.data_file + '.png')
    plt.savefig(args.out_fig + args.data_file + '.eps', format = 'eps')
    plt.show()

if __name__ == '__main__':
    main()
