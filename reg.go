package flyml

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type LogIt struct {
	Label     map[string]int
	LabelVals []float64
	Future    map[int]int
	Weights   []float64
	Rate      float64
}

// TODO: add logloss info?
func logloss(yTrue float64, yPred float64) float64 {
	loss := yTrue*math.Log1p(yPred) + (1-yTrue)*math.Log1p(1-yPred)
	return loss
}

// LogItNew create new LogIt with learning rate (0..1)
func LogItNew(learningRate float64) *LogIt {
	return &LogIt{
		Label:     make(map[string]int),
		LabelVals: make([]float64, 0),
		Future:    make(map[int]int),
		Weights:   make([]float64, 0),
		Rate:      learningRate,
	}
}

// LabelPut add new Label or return label index
func (li *LogIt) LabelPut(label string) (idx int, isNew bool) {
	idx, ok := li.Label[label]
	if !ok {
		li.Label[label] = len(li.Label)
		li.LabelVals = li.LabelOnehot()
		return li.Label[label], true
	}
	return idx, false
}

// LabelOnehot label 2 dictionary
func (li *LogIt) LabelOnehot() []float64 {
	v := mat.NewVecDense(len(li.Label), nil)
	// make labels slice
	labels := make([]string, 0, len(li.Label))
	for key := range li.Label {
		labels = append(labels, key)
	}
	// sort by value ascending
	sort.Slice(labels, func(i, j int) bool {
		return li.Label[labels[i]] < li.Label[labels[j]]
	})
	//fmt.Println(labels)

	for i := 0; i < len(labels); i++ {
		//fmt.Println("i", i)
		f, ok := li.Label[labels[i]]
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
	idx, ok := li.Future[futureHash]
	if !ok {
		li.Future[futureHash] = len(li.Future)
		return len(li.Future), true
	}
	return idx, false
}

//LoadLineSVM convert line in svm format:
//1 6:1 8:1 15:1 21:1 29:1 33:1 34:1 37:1 42:1 50:1
//Label FutureHash:FutureWeight ...
func (li *LogIt) LoadLineSVM(s string) (labelID int, futures []float64, err error) {

	fields := strings.Fields(s)
	futures = make([]float64, len(li.Future))
	for i := range fields {
		if i == 0 {
			labelID, _ = li.LabelPut(fields[0])
			continue //label
		}
		arr := strings.Split(fields[i], ":")
		if len(arr) < 2 {
			return labelID, futures, fmt.Errorf("Error in string:%s len(arr) < 2", s)
		}
		futureHash, err := strconv.Atoi(arr[0])
		if err != nil {
			return labelID, futures, fmt.Errorf("Error in string:%s err:%s", s, err)
		}
		futureVal, err := strconv.ParseFloat(arr[1], 64)
		if err != nil {
			return labelID, futures, fmt.Errorf("Error in string:%s err:%s", s, err)
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

// Softmax return absolute probability value
func Softmax(w, x *mat.VecDense) float64 {
	v := mat.Dot(w, x)
	return 1.0 / (1.0 + math.Exp(-v))
}

// LineTrain train one line with gradient descent
func (li *LogIt) LineTrain(futVals []float64, labelVal float64) {
	//fmt.Println("fm3", len(futVals))
	// vectorize
	if len(futVals) != len(li.Weights) {
		if len(futVals) > len(li.Weights) {
			// new futures
			tmp := make([]float64, len(futVals)-len(li.Weights))
			li.Weights = append(li.Weights, tmp...)
		}
	}
	x := mat.NewVecDense(len(futVals), futVals)
	wsVec := mat.NewVecDense(len(li.Weights), li.Weights)
	// predict
	pred := Softmax(wsVec, x)
	// predict error
	predErr := labelVal - pred
	// scale koef
	scale := li.Rate * predErr * pred * (1 - pred)
	// calc scaled gradient
	dx := mat.NewVecDense(x.Len(), nil)
	dx.CopyVec(x)
	//TODO
	//try GD from https://pythobyte.com/logistic-regression-from-scratch-ae373d5d/
	//np.dot(X.T, (h - y)) / y.shape[0]
	// self.weight -= lr * dW
	dx.ScaleVec(scale, x) //*float64(x.Len()), x)
	// apply gradient
	wsVec.AddVec(wsVec, dx)
	li.Weights = wsVec.RawVector().Data
}

// TrainLineSVM train singl line
func (li *LogIt) TrainLineSVM(s string) []float64 {
	labelIdx, futures, err := li.LoadLineSVM(s)
	if err != nil {
		log.Fatal(err)
	}
	if len(li.Weights) == 0 {
		li.Weights = make([]float64, len(futures))
	}
	li.LineTrain(futures, li.LabelVals[labelIdx])
	return li.Weights
}

// TrainLinesSVM train on batch lines
// Stohastic Gradient Descent
func (li *LogIt) TrainLinesSVM(strs []string, epoch int) {
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
	li.WarmUp(futures, labels, epoch)
}

// WarmUp regression with classic interface
func (li *LogIt) WarmUp(futVals [][]float64, labelVal []float64, epoch int) []float64 {
	for i := range futVals {
		if len(futVals[i]) < len(li.Future) {
			tmp := make([]float64, len(li.Future)-len(futVals[i]))
			futVals[i] = append(futVals[i], tmp...)
		}
	}

	for ep := 0; ep < epoch; ep++ {
		//shuffle
		rand.NewSource(int64(ep))
		rand.Shuffle(len(futVals), func(i, j int) {
			futVals[i], futVals[j] = futVals[j], futVals[i]
			labelVal[i], labelVal[j] = labelVal[j], labelVal[i]
		})
		for i := range futVals {
			li.LineTrain(futVals[i], labelVal[i])
		}
	}
	return li.Weights
}

// TestLinesSVM test multiple lines
// return accuracy
func (li *LogIt) TestLinesSVM(strs []string) float64 {
	labels := make([]float64, len(strs))
	futures := make([][]float64, len(strs))

	var wrong int
	for i, s := range strs {
		labelIdx, future, err := li.LoadLineSVM(s)
		if err != nil {
			log.Fatal(err)
		}
		labels[i] = li.LabelVals[labelIdx]
		futures[i] = future
		_, _, labelPredIdx := li.Predict(future)
		if labelPredIdx != labelIdx {
			wrong++
		}
	}
	return 100 * (1 - float64(wrong)/float64(len(strs)))
}

// Predict
func (li *LogIt) Predict(futures []float64) (probability float64, label string, labelIdx int) {
	labels := make([]string, 0, len(li.Label))
	for key := range li.Label {
		labels = append(labels, key)
	}
	// sort by value ascending
	sort.Slice(labels, func(i, j int) bool {
		return li.Label[labels[i]] < li.Label[labels[j]]
	})
	pred := Softmax(mat.NewVecDense(len(li.Weights), li.Weights), mat.NewVecDense(len(futures), futures))
	min := 1.
	num := 0
	for k, v := range li.LabelVals {
		if math.Abs(pred-v) <= min {
			min = math.Abs(pred - v)
			num = k
		}
	}
	return pred, labels[num], li.Label[labels[num]]
}
