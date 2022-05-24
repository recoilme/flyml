package flyml

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type LogIt struct {
	sync.Mutex
	LabelIdx  map[string]int
	Labels    []string
	LabelVals []float64
	Future    map[int]int
	Features  []float64
	Weights   []float64
	Rate      float64
}

// logloss
func logloss(yTrue float64, yPred float64) float64 {
	loss := yTrue*math.Log1p(yPred) + (1-yTrue)*math.Log1p(1-yPred)
	return loss
}

// LogItNew create new LogIt with learning rate (0..1)
func LogItNew(learningRate float64, seed int) *LogIt {
	rand.Seed(int64(seed))
	return &LogIt{
		LabelIdx:  make(map[string]int),
		LabelVals: make([]float64, 0),
		Labels:    make([]string, 0),
		Future:    make(map[int]int),
		Features:  make([]float64, 0),
		Weights:   make([]float64, 0),
		Rate:      learningRate,
	}
}

func (li *LogIt) CleanFeatures() {
	for i, _ := range li.Features {
		li.Features[i] = 0.0
	}
}

// LabelPut add new Label or return label index
func (li *LogIt) LabelPut(label string) (idx int, isNew bool) {
	li.Lock()
	defer li.Unlock()
	idx, ok := li.LabelIdx[label]
	if !ok {
		li.LabelIdx[label] = len(li.LabelIdx)
		li.Labels = append(li.Labels, label)
		li.LabelVals = li.LabelOnehot(li.Labels)
		return li.LabelIdx[label], true
	}
	return idx, false
}

func (li *LogIt) LabelOnehot(labels []string) []float64 {
	v := mat.NewVecDense(len(labels), nil)
	for i := 0; i < len(labels); i++ {
		//fmt.Println("i", i)
		f, ok := li.LabelIdx[labels[i]]
		if ok {
			//fmt.Println(i, f)
			v.SetVec(i, float64(f))
		}
	}
	v.ScaleVec(1/float64(len(labels)), v)

	return v.RawVector().Data
}

// FuturePut add FutureHash to dictionary or return index
func (li *LogIt) FuturePut(futureHash int) (idx int, isNew bool) {
	li.Lock()
	defer li.Unlock()
	idx, ok := li.Future[futureHash]
	if !ok {
		li.Future[futureHash] = len(li.Future)
		li.Weights = append(li.Weights, rand.Float64()) //(rand.Float64()-0.5)*float64(len(li.Future)/2))
		return len(li.Future), true
	}
	return idx, false
}

//LoadLineSVM convert line in svm format:
//1 6:1 8:1 15:1 21:1 29:1 33:1 34:1 37:1 42:1 50:1
//Label FutureHash:FutureWeight ...
//Wight can be ommited
func (li *LogIt) LoadLineSVM(s string) (labelID int, futures []float64, err error) {

	fields := strings.Fields(s)
	futures = make([]float64, len(li.Future))
	var futureVal float64
	for i := range fields {
		if i == 0 {
			labelID, _ = li.LabelPut(fields[0])
			continue //label
		}
		arr := strings.Split(fields[i], ":")
		futureHash, err := strconv.Atoi(arr[0])
		if err != nil {
			return labelID, futures, fmt.Errorf("Error in string:%s err:%s", s, err)
		}

		if len(arr) < 2 {
			futureVal = 1.0
		} else {
			futureVal, err = strconv.ParseFloat(arr[1], 64)
			if err != nil {
				return labelID, futures, fmt.Errorf("Error in string:%s err:%s", s, err)
			}
		}

		futureIdx, isNew := li.FuturePut(futureHash)
		if isNew {
			//new future
			futures = append(futures, futureVal)
			continue
		}

		futures[futureIdx] = futureVal
	}
	return labelID, futures, nil
}

// Softmax
func Softmax(x, w *mat.VecDense) float64 {
	v := mat.Dot(x, w)
	return 1.0 / (1.0 + math.Exp(-v))
}

// Train train one line with gradient descent
func (li *LogIt) Train(futVals []float64, labelVal float64, calcLoss bool) (loss float64) {
	// vectorize
	li.Lock()
	if len(futVals) != len(li.Weights) {
		if len(futVals) > len(li.Weights) {
			// new futures
			tmp := make([]float64, len(futVals)-len(li.Weights))
			li.Weights = append(li.Weights, tmp...)
		}
	}
	x := mat.NewVecDense(len(futVals), futVals)
	wsVec := mat.NewVecDense(len(li.Weights), li.Weights)
	li.Unlock()
	// predict
	pred := Softmax(x, wsVec)

	if calcLoss {
		loss = logloss(labelVal, pred)
	}

	//Linear Approximation
	//for i := range li.Weights {
	//li.Weights[i] += li.Rate * ((labelVal - pred) * futVals[i])
	//}
	scale := li.Rate * (labelVal - pred)
	wsVec.AddScaledVec(wsVec, scale, x)
	li.Lock()
	li.Weights = wsVec.RawVector().Data
	li.Unlock()
	//TODO: other solvers https://stackoverflow.com/questions/38640109/logistic-regression-python-solvers-definitions
	return
	/*

		// error
		predErr := labelVal - pred
		// scale koef
		scale := li.Rate * predErr * (pred * (1 - pred))

		for i := range li.Weights {
			li.Weights[i] = li.Weights[i] + scale*futVals[i]
		}
		return*/

	/*
		dx := mat.NewVecDense(len(futVals), nil)
		dx.CopyVec(x)
		dx.ScaleVec(scale, x)
		wsVec.AddVec(wsVec, dx)
		li.Weights = wsVec.RawVector().Data
		fmt.Println(li.Weights[0], dx.RawVector().Data[0])
		return loss
	*/

}

// TrainLine train single line
func (li *LogIt) TrainLine(s string) (err error) {
	var featureVal float64
	var labelID int
	var labelVal float64
	li.CleanFeatures()
	fields := strings.Fields(s)
	for i := range fields {
		if i == 0 {
			labelID, _ = li.LabelPut(fields[0])
			labelVal = li.LabelVals[labelID]
			continue //label
		}
		arr := strings.Split(fields[i], ":")
		featureHash, err := strconv.Atoi(arr[0])
		if err != nil {
			return fmt.Errorf("Error in string:%s err:%s", s, err)
		}

		featureVal = 1.0
		if len(arr) > 2 {
			featureVal, err = strconv.ParseFloat(arr[1], 64)
			if err != nil {
				return fmt.Errorf("Error in string:%s err:%s", s, err)
			}
		}
		featureIdx, isNew := li.FuturePut(featureHash)
		if isNew {
			li.Features = append(li.Features, featureVal)
			continue
		}
		li.Features[featureIdx] = featureVal
	}
	// vectorize
	li.Lock()
	if len(li.Features) != len(li.Weights) {
		if len(li.Features) > len(li.Weights) {
			// new futures
			tmp := make([]float64, len(li.Features)-len(li.Weights))
			li.Weights = append(li.Weights, tmp...)
		}
	}
	wsVec := mat.NewVecDense(len(li.Weights), li.Weights)
	x := mat.NewVecDense(len(li.Features), li.Features)
	li.Unlock()
	// predict
	pred := Softmax(x, wsVec)
	scale := li.Rate * (labelVal - pred)
	wsVec.AddScaledVec(wsVec, scale, x)
	li.Lock()
	li.Weights = wsVec.RawVector().Data
	li.Unlock()
	return nil
}

// TrainLines train on batch lines
// Stohastic Gradient Descent
func (li *LogIt) TrainLines(strs []string, epoch int) {
	if epoch <= 0 {
		return
	}
	labels := make([]float64, len(strs))
	futures := make([][]float64, len(strs))
	for i, s := range strs {
		labelIdx, future, err := li.LoadLineSVM(s)
		if err != nil {
			log.Fatal(err)
		}
		labels[i] = li.LabelVals[labelIdx]
		futures[i] = future
	}
	li.WarmUp(futures, labels, strs, epoch)
}

// WarmUp regression with classic interface
func (li *LogIt) WarmUp(futVals [][]float64, labelVal []float64, test []string, epoch int) []float64 {
	for i := range futVals {
		if len(futVals[i]) < len(li.Future) {
			tmp := make([]float64, len(li.Future)-len(futVals[i]))
			futVals[i] = append(futVals[i], tmp...)
		}
	}

	for ep := 0; ep < epoch; ep++ {
		//shuffle
		rand.Seed(int64(ep))
		rand.Shuffle(len(futVals), func(i, j int) {
			futVals[i], futVals[j] = futVals[j], futVals[i]
			labelVal[i], labelVal[j] = labelVal[j], labelVal[i]
		})
		if ep%50 == 0 || ep == (epoch-1) {
			loss := 0.
			for i := range futVals {
				loss += li.Train(futVals[i], labelVal[i], true)
			}
			fmt.Printf("ep:%d loss:%f accuracy:%f\n", ep, loss/float64(len(futVals)), li.TestLinesSVM(test))
			continue
		}
		var wg sync.WaitGroup
		for i := range futVals {
			wg.Add(1)
			go func(x []float64, y float64) {
				defer wg.Done()
				li.Train(x, y, false)
			}(futVals[i], labelVal[i])
		}
		wg.Wait()

	}
	return li.Weights
}

// TestLinesSVM test multiple lines
// return accuracy
func (li *LogIt) TestLinesSVM(strs []string) float64 {
	//labels := make([]float64, len(li.Labels))
	//futures := make([][]float64, len(strs))

	var wrong int
	for i, s := range strs {
		labelIdx, future, err := li.LoadLineSVM(s)
		if err != nil {
			log.Fatal(err)
		}
		//labels[i] = li.LabelVals[labelIdx]
		//futures[i] = future
		prob, label, labelPredIdx := li.Predict(future)
		if i < 0 {
			fmt.Println(s, "\n", prob, label, labelPredIdx, labelIdx)
		}
		if labelPredIdx != labelIdx {
			wrong++
		}
	}
	return 100 * (1 - float64(wrong)/float64(len(strs)))
}

// Predict
func (li *LogIt) Predict(futures []float64) (probability float64, label string, labelIdx int) {

	pred := Softmax(mat.NewVecDense(len(futures), futures), mat.NewVecDense(len(li.Weights), li.Weights))
	min := 1.
	num := 0
	for k, v := range li.LabelVals {
		if math.Abs(pred-v) <= min {
			min = math.Abs(pred - v)
			num = k
		}
	}
	return pred, li.Labels[num], li.LabelIdx[li.Labels[num]]
}
