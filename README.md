# GFM-Plus

## Nonconvex SVM

To reproduce the experiments, run

```
cd ./SVM
python -u train.py --data_file a9a/w8a/covtype/ijcnn --epochs=10
python -u train.py --data_file mushrooms --epochs=50
python -u train.py --data_file phishing --epochs=200
```

The datasets used in our experiments are available at https://www.csie.ntu.edu.tw/~cjlin/libsvm/

## Black-Box Attack on CNN

To reproduce the experiments on MNIST, please download the MNIST dataset in `args.data_dir` and run
```
cd ./Attack
python -u AttackMNIST.py
```
You can also download the dataset automatically by setting  `download=True` in the following code
```
datasets.MNIST(root=args.data_dir, train=False, transform=test_transform, download=True)
```
The experiments on FashionMNIST are very similar and we do not contain the codes here.

