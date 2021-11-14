# FlyML perfomant real time mashine learning libraryes in Go

## simple & perfomant logistic regression (~100 LoC)

## Status: WIP!

Validated on mushrooms dataset

```
Starting adding futures from < dataset/mushrooms.svm >
        Futures: 112
        Average iter time: 578ns
Finished Testing < logistic regression >
        Accuracy: 100 percent
        Examples tested: 1625
        Average Classification Time: 342ns
Starting adding futures from < dataset/mushrooms.svm >
        Futures: 112
```

sklearn give ~94.9% accuracy: https://medium.com/analytics-vidhya/mushroom-classification-using-different-classifiers-aa338c1cd0ff

With 50 epochs accuracy: 100 percent

## Usage

see tests & examples

## Credits

https://github.com/mattn/go-gonum-logisticregression-iris
https://github.com/haydenhigg/logan