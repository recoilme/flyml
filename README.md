# FlyML 
perfomant real time mashine learning libraryes in Go

## simple & perfomant logistic regression (~300 LoC)

## Status: WIP!

Validated on mushrooms dataset

```
Finished Testing < logistic regression mashroom >
        Accuracy (online learn): 79.58

ep:0 loss:0.513353 accuracy:87.838503 w[0];-0.299532
ep:50 loss:0.540091 accuracy:99.729197 w[0];-0.178849
ep:100 loss:0.541736 accuracy:99.901526 w[0];-0.199327
ep:150 loss:0.542391 accuracy:99.901526 w[0];-0.192892
ep:200 loss:0.542744 accuracy:99.901526 w[0];-0.180376
ep:250 loss:0.542968 accuracy:99.901526 w[0];-0.167066
ep:300 loss:0.543120 accuracy:99.913836 w[0];-0.153703
ep:350 loss:0.543234 accuracy:99.913836 w[0];-0.140660
ep:400 loss:0.543322 accuracy:99.926145 w[0];-0.128059
ep:450 loss:0.543389 accuracy:99.926145 w[0];-0.116019
ep:500 loss:0.543446 accuracy:99.926145 w[0];-0.104360
        Average iter time: 737ns
        Futures: 112 Labels: 2
        Average prediction time: 3.232µs
        Accuracy (offline): 99.88


Finished Testing < logistic regression: iris >
        Accuracy (online learn/1 epoch): 66.67

        map[Iris-setosa:1 Iris-versicolor:2 Iris-virginica:0]   [0 0.3333333333333333 0.6666666666666666]    [Iris-virginica Iris-setosa Iris-versicolor]
ep:0 loss:0.407010 accuracy:33.333333 w[0];-0.193984
ep:50 loss:0.444028 accuracy:33.333333 w[0];-0.084339
ep:100 loss:0.446542 accuracy:33.333333 w[0];-0.175733
ep:150 loss:0.446536 accuracy:33.333333 w[0];-0.267623
ep:200 loss:0.448969 accuracy:34.074074 w[0];-0.345329
ep:250 loss:0.447641 accuracy:37.037037 w[0];-0.415924
ep:300 loss:0.450048 accuracy:37.777778 w[0];-0.478445
ep:350 loss:0.448828 accuracy:37.777778 w[0];-0.533022
ep:400 loss:0.449873 accuracy:39.259259 w[0];-0.581665
ep:450 loss:0.449983 accuracy:39.259259 w[0];-0.624211
ep:500 loss:0.453216 accuracy:37.777778 w[0];-0.657195
        Average iter time: 320ns
        Futures: 4 Labels: 3
        Average prediction time: 878ns
        Accuracy (offline 501 epoch): 33.33


Finished Testing < logistic regression: breast-cancer >
        Accuracy (online learn/1 epoch): 95.65

        map[2:0 4:1]    [0 0.5] [2 4]
ep:0 loss:0.576339 accuracy:92.345277 w[0];0.079586
ep:50 loss:0.576080 accuracy:93.485342 w[0];0.403783
ep:100 loss:0.576898 accuracy:93.648208 w[0];0.578743
ep:150 loss:0.577313 accuracy:93.811075 w[0];0.670421
ep:200 loss:0.577598 accuracy:93.811075 w[0];0.721227
ep:250 loss:0.577750 accuracy:93.811075 w[0];0.751007
ep:300 loss:0.577836 accuracy:93.811075 w[0];0.769067
ep:350 loss:0.577943 accuracy:93.811075 w[0];0.780843
ep:400 loss:0.577987 accuracy:93.811075 w[0];0.788499
ep:450 loss:0.578018 accuracy:93.811075 w[0];0.793840
ep:500 loss:0.578055 accuracy:93.811075 w[0];0.797460
        Average iter time: 308ns
        Futures: 10 Labels: 2
        Average prediction time: 1.762µs
        Accuracy (offline 501 epoch): 94.20


Finished Testing < logistic regression: news20 >
        Accuracy (online learn/1 epoch): 5.33

ep:0 loss:0.367668 accuracy:6.952095 w[0];-0.636263
ep:50 loss:0.455879 accuracy:11.930828 w[0];-1.438584
ep:100 loss:0.468886 accuracy:13.381215 w[0];-1.290299
ep:150 loss:0.475668 accuracy:14.985008 w[0];-1.177516
ep:200 loss:0.479675 accuracy:16.735235 w[0];-1.112868
ep:250 loss:0.482491 accuracy:18.283244 w[0];-1.069842
ep:300 loss:0.484196 accuracy:19.391953 w[0];-1.045669
ep:350 loss:0.485600 accuracy:20.939962 w[0];-0.998757
ep:400 loss:0.486857 accuracy:22.229970 w[0];-0.983233
ep:450 loss:0.487937 accuracy:24.426470 w[0];-0.963490
ep:500 loss:0.488566 accuracy:25.521233 w[0];-0.940698
        Average iter time: 130.039µs
        Futures: 60346 Labels: 20
        Average prediction time: 113.23µs
        Accuracy (offline 501 epoch): 10.41

```


## Usage

see tests & examples

```
	filepath := "dataset/mushrooms.svm"
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)
        
        li := flyml.LogItNew(0.1)
	for scanner.Scan() {
		scanText := scanner.Text()
		li.TrainLineSVM(scanText)
	}
```

## Credits

https://github.com/mattn/go-gonum-logisticregression-iris