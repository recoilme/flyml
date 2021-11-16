# FlyML 
perfomant real time mashine learning libraryes in Go

## simple & perfomant logistic regression (~300 LoC)

## Status: WIP!

Validated on mushrooms dataset

```
        Accuracy (online learn/1 epoch): 99.26

        Average iter time: 2.208µs
        Futures: 112 Labels: 2
        Average prediction time: 4.218µs
        Accuracy (offline/3 epoch): 100.00
```

[sklearn give ~94.9% accuracy (max_iter=500)](https://medium.com/analytics-vidhya/mushroom-classification-using-different-classifiers-aa338c1cd0ff)


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