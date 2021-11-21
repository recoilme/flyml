# FlyML 
perfomant real time mashine learning libraryes in Go

## simple & perfomant logistic regression

### Description
 - Method: Stohastic Gradient Descent
 - Solver: Linear Approximation
 - Format: SVM data format, datasets: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

### Advanages

 - Sparse data, mix online/offline learn. May be used "on the fly"
 - Perfomant, based on gonum (gonum support BLAS https://githubhelp.com/gonum/blas)
 - Seed for predictable results
 - Validated on different toys datasets
 - Logloss, accuracy metrics
 - Parallel learning

### Disadvantages:
 
 - slow learning on multilabeled datasets (see tests)


## Status: WIP!

## Validation

Validated on toys datasets

```
Finished Testing < logistic regression mashroom >
        Accuracy (online learn): 79.58

ep:0 loss:0.513353 accuracy:87.838503
ep:50 loss:0.540039 accuracy:99.729197
ep:100 loss:0.541703 accuracy:99.901526
ep:150 loss:0.542366 accuracy:99.901526
ep:200 loss:0.542724 accuracy:99.901526
ep:250 loss:0.542951 accuracy:99.901526
ep:300 loss:0.543106 accuracy:99.913836
ep:350 loss:0.543222 accuracy:99.913836
ep:400 loss:0.543310 accuracy:99.926145
ep:450 loss:0.543378 accuracy:99.926145
ep:500 loss:0.543436 accuracy:99.926145
        Average iter time: 927ns
        Futures: 112 Labels: 2
        Average prediction time: 3.817µs
        Accuracy (offline): 99.88

150

Finished Testing < logistic regression: iris >
        Accuracy (online learn): 66.67

ep:0 loss:0.407010 accuracy:33.333333
ep:50 loss:0.445142 accuracy:33.333333
ep:100 loss:0.446651 accuracy:33.333333
ep:150 loss:0.447010 accuracy:33.333333
ep:200 loss:0.447819 accuracy:34.074074
ep:250 loss:0.447078 accuracy:37.037037
ep:300 loss:0.450236 accuracy:37.777778
ep:350 loss:0.448991 accuracy:37.777778
ep:400 loss:0.449492 accuracy:39.259259
ep:450 loss:0.448452 accuracy:39.259259
ep:500 loss:0.452250 accuracy:37.037037
        Average iter time: 1.289µs
        Futures: 4 Labels: 3
        Average prediction time: 1.83µs
        Accuracy (offline 501 epoch): 33.33


Finished Testing < logistic regression: breast-cancer >
        Accuracy (online learn): 95.65

ep:0 loss:0.576339 accuracy:92.345277
ep:50 loss:0.576001 accuracy:93.485342
ep:100 loss:0.576893 accuracy:93.648208
ep:150 loss:0.577251 accuracy:93.811075
ep:200 loss:0.577570 accuracy:93.811075
ep:250 loss:0.577755 accuracy:93.811075
ep:300 loss:0.577802 accuracy:93.811075
ep:350 loss:0.577912 accuracy:93.811075
ep:400 loss:0.577938 accuracy:93.811075
ep:450 loss:0.578021 accuracy:93.811075
ep:500 loss:0.578071 accuracy:93.811075
        Average iter time: 942ns
        Futures: 10 Labels: 2
        Average prediction time: 7.118µs
        Accuracy (offline 501 epoch): 94.20

Finished Testing < logistic regression: news20 >
        Accuracy (online learn): 5.33

ep:0 loss:0.367668 accuracy:6.952095
ep:50 loss:0.455712 accuracy:11.665853
ep:100 loss:0.468759 accuracy:13.583432
ep:150 loss:0.475536 accuracy:14.831602
ep:200 loss:0.479495 accuracy:16.526044
ep:250 loss:0.482260 accuracy:17.983404
ep:300 loss:0.484181 accuracy:19.419845
ep:350 loss:0.485859 accuracy:20.709853
ep:400 loss:0.486729 accuracy:22.620459
ep:450 loss:0.487848 accuracy:24.426470
ep:499 loss:0.488426 accuracy:24.405551
        Average iter time: 41.226µs
        Total time: 4m55.617333242s
        Futures: 60346 Labels: 20
        Average prediction time: 120.246µs
        Accuracy (offline 500 epoch): 10.41
```


## Usage

see tests

## Credits

https://github.com/mattn/go-gonum-logisticregression-iris