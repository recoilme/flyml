package flyml

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
	Means   []float64 `json:"means"`
	Rand    *rand.Rand
}

func (mdl *Model) TrainSGD(inputs [][]float64, outputs []float64, epochs int) {
	mdl.calculateMeans(inputs)

	m := len(inputs)

	if len(mdl.Weights) == 0 {
		mdl.Weights = make([]float64, len(inputs[0]))
	}

	var index int
	epochs *= len(outputs)

	for epoch := 0; epoch < epochs; epoch++ {
		index = mdl.Rand.Intn(m)
		//index = epoch
		mdl.train(inputs[index], outputs[index])
		//if epoch%len(outputs) == 0 {
		//	fmt.Println(epoch, mdl.Accuracy(inputs, outputs), index)
		//}
	}
}

func (mdl *Model) calculateMeans(inputs [][]float64) {
	mdl.Means = make([]float64, len(inputs[0]))

	for i := range inputs {
		for j, feature := range inputs[i] {
			mdl.Means[j] += feature
		}
	}

	m := float64(len(inputs))

	for j := range mdl.Means {
		mdl.Means[j] /= m
	}
}

func (mdl *Model) train(input []float64, output float64) {
	scaled := subtract(input, mdl.Means)
	delta := mdl.predict(scaled) - output

	for j, feature := range scaled {
		mdl.Weights[j] -= delta * feature
	}

	mdl.Bias -= delta
}

func (mdl *Model) predict(input []float64) float64 {
	return sigmoid(dot(input, mdl.Weights) + mdl.Bias)
}

func (mdl *Model) Predict(input []float64) float64 {
	return mdl.predict(subtract(input, mdl.Means))
}

func subtract(a, b []float64) []float64 {
	/*
		difference := make([]float64, len(a))

		for i, x := range a {
			difference[i] = x - b[i]
		}

		return difference
	*/
	vec := mat.NewVecDense(len(a), a)
	vec.SubVec(vec, mat.NewVecDense(len(b), b))
	return vec.RawVector().Data
}

func dot(a, b []float64) float64 {
	/*
		s := 0.

		for i, x := range a {
			s += x * b[i]
		}

		return s
	*/
	return mat.Dot(mat.NewVecDense(len(a), a), mat.NewVecDense(len(b), b))
}

func sigmoid(z float64) float64 {
	return 1. / (1. + math.Exp(-z))
}

func (mdl *Model) Accuracy(test [][]float64, testY []float64) float64 {
	var count int
	var wrong int

	for i := range test {
		guess := mdl.Predict(test[i])
		if (guess < .5 && testY[i] == 1) || (guess > .5 && testY[i] == 0) { //abs(guess[0]-testY[i]) > 1e-2 {
			wrong++
		}
		count++
	}
	return 100 * (1 - float64(wrong)/float64(count))
}

func LogisticRegression(xx [][]float64, yy []float64, rate float64, ntrains int) *mat.VecDense {
	X := make([]*mat.VecDense, len(xx))
	for i := 0; i < len(X); i++ {
		X[i] = mat.NewVecDense(len(xx[i]), xx[i])
	}
	y := mat.NewVecDense(len(yy), yy)
	ws := make([]float64, X[0].Len())
	//for i := range ws {
	//	ws[i] = (rand.Float64() - 0.5) * float64(X[0].Len()/2)
	//}
	//fmt.Println(X[0].RawVector().Data)
	w := mat.NewVecDense(len(ws), ws)
	for n := 0; n < ntrains; n++ {
		for i, x := range X {
			//t := mat.NewVecDense(x.Len(), nil)
			//t.CopyVec(x)
			pred := softmax(w, x)
			perr := y.At(i, 0) - pred
			scale := rate * perr * pred * (1 - pred)
			dx := mat.NewVecDense(x.Len(), nil)
			dx.CopyVec(x)
			//dx := mat.NewVecDense(x.Len(), x.RawVector().Data)
			dx.ScaleVec(scale, x)
			//for j := 0; j < x.Len(); j++ {
			w.AddVec(w, dx)
			//}
		}
	}
	//fmt.Println(X[0].RawVector().Data)
	return w
}

func softmax(w, x *mat.VecDense) float64 {
	v := mat.Dot(w, x)
	return 1.0 / (1.0 + math.Exp(-v))
}

func predictsm(w, x []float64) float64 {
	return softmax(mat.NewVecDense(len(w), w), mat.NewVecDense(len(x), x))
}

func Accuracy(test [][]float64, testY, w []float64) float64 {
	var count int
	var wrong int

	for i := range test {
		guess := predictsm(w, test[i])
		if (guess < .5 && testY[i] == 1) || (guess > .5 && testY[i] == 0) { //abs(guess[0]-testY[i]) > 1e-2 {
			wrong++
		}
		//fmt.Printf("%f:%f\n", guess, testY[i])
		count++
	}
	return 100 * (1 - float64(wrong)/float64(count))
}
