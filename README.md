# GFM_Plus

Codes for ``Faster Gradient-Free Algorithms for Nonsmooth Nonconvex Stochastic Optimization'' in ICML 2023

## Nonconvex SVM
To reproduce the experments of nonconvex SVM, run

```
python -u train.py --data_file a9a/w8a/covtype/ijcnn --epochs=10
python -u train.py --data_file mushrooms --epochs=50
python -u train.py --data_file phishing --epochs=200
```

The datasets used in our experiments are available at https://www.csie.ntu.edu.tw/~cjlin/libsvm/

## Black-Box Attack on CNN

