package flyml

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type LogIt struct {
	Label     map[string]int
	Labels    []string
	LabelVals []float64
	Future    map[int]int
	Weights   []float64
	Rate      float64
}

// TODO: add logloss info?
func logloss(yTrue float64, yPred float64) float64 {
	loss := yTrue*math.Log(yPred+0.00001) + (1-yTrue)*math.Log(1-yPred)
	return loss
}

// LogItNew create new LogIt with learning rate (0..1)
func LogItNew(learningRate float64) *LogIt {
	return &LogIt{
		Label:     make(map[string]int),
		LabelVals: make([]float64, 0),
		Labels:    make([]string, 0),
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
		li.Labels = append(li.Labels, label)
		li.LabelVals = li.LabelOnehot()
		return li.Label[label], true
	}
	return idx, false
}

func (li *LogIt) LabelOnehot() []float64 {
	if len(li.Label) == 2 {
		return []float64{0.000001, 0.999999}
	}
	if len(li.Label) == 3 {
		return []float64{0.000001, .555555, 0.999999}
	}
	v := mat.NewVecDense(len(li.Label), nil)
	for i := 0; i < len(li.Labels); i++ {
		//fmt.Println("i", i)
		f, ok := li.Label[li.Labels[i]]
		if ok {
			//fmt.Println(i, f)
			v.SetVec(i, float64(f))
		}
	}
	v.ScaleVec(1/float64(len(li.Labels)), v)

	return v.RawVector().Data
}

// FuturePut add FutureHash to dictionary or return index
func (li *LogIt) FuturePut(futureHash int) (idx int, isNew bool) {
	idx, ok := li.Future[futureHash]
	if !ok {
		li.Future[futureHash] = len(li.Future)
		li.Weights = append(li.Weights, 0.0)
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

// Softmax
func Softmax(x, w *mat.VecDense) float64 {
	v := mat.Dot(x, w)
	return 1.0 / (1.0 + math.Exp(-v))
}

// Train train one line with gradient descent
func (li *LogIt) Train(futVals []float64, labelVal float64) (loss float64) {
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
	pred := Softmax(x, wsVec)

	loss = logloss(pred, labelVal)

	for i := range li.Weights {
		li.Weights[i] += li.Rate * ((labelVal - pred) * futVals[i])
	}
	return
	//dx := mat.NewVecDense(x.Len(), nil)
	//dx.CopyVec(x)
	//dx.ScaleVec(predErr, x)

	//dx.SubVec(dx, labelVal)
	//wsVec.SubVec((labelVal - dx), wsVec)
	//li.Weights = wsVec.RawVector().Data
}

// TrainLine train single line
func (li *LogIt) TrainLine(s string) []float64 {
	labelIdx, futures, err := li.LoadLineSVM(s)
	if err != nil {
		log.Fatal(err)
	}
	li.Train(futures, li.LabelVals[labelIdx])
	return li.Weights
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
		loss := 0.
		for i := range futVals {
			loss += li.Train(futVals[i], labelVal[i])
		}
		if ep%50 == 0 {
			fmt.Printf("ep:%d loss:%f\n", ep, loss/float64(len(futVals)))
		}
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
	//fmt.Println(labels[num])
	//fmt.Println(pred, int(float64(len(li.Label))*pred+0.1), labels[num], li.Label[labels[num]], futures)
	return pred, li.Labels[num], li.Label[li.Labels[num]]

}