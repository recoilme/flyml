package flyml_test

import (
	"archive/zip"
	"bufio"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/recoilme/flyml"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/blas/blas64"
)

func ShuffleStr(slice []string, seed int) []string {
	rand.Seed(int64(seed))
	rand.Shuffle(len(slice), func(i, j int) {
		slice[i], slice[j] = slice[j], slice[i]
	})
	return slice
}

func TestSVM(t *testing.T) {
	li := flyml.LogItNew(0.1, 42)
	_, _, err := li.LoadLineSVM("1 6:1 8:1 15:1 21:1 29:1 33:1 34:1 37:1 42:1 50:1")
	assert.NoError(t, err)
	_, fut, err := li.LoadLineSVM("2 7:1 8:1")
	assert.NoError(t, err)
	fmt.Printf("%+v\n", li)
	// &{Label:map[1:0 2:1]
	// Future:map[6:0 7:10 8:1 15:2 21:3 29:4 33:5 34:6 37:7 42:8 50:9]}
	// [0 1 0 0 0 0 0 0 0 0 1]
	assert.Equal(t, 11, len(fut))
	assert.Equal(t, 2, len(li.LabelIdx))
}

func TestMashroom(t *testing.T) {
	seed := 42
	filepath := "dataset/mushrooms.svm"
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	lines := make([]string, 0, 1024)
	for scanner.Scan() {
		scanText := scanner.Text()
		lines = append(lines, scanText)
	}

	// shuffle dataset
	lines = ShuffleStr(lines, seed)

	// separate training and test sets
	trainLen := int(0.9 * float64(len(lines)))
	train := lines[:trainLen]

	li := flyml.LogItNew(0.001, seed)
	// online learn & load
	loss := 0.
	test := lines[trainLen:]
	for i, s := range train {
		labelIdx, futures, err := li.LoadLineSVM(s)
		if err != nil {
			log.Fatal(err)
		}
		loss += li.Train(futures, li.LabelVals[labelIdx], true)
		lossTest := 0.
		for _, sTest := range test {
			lossTest += li.TestLineLogLoss(sTest)
		}
		fmt.Printf("ep:%d loss:%f lossTest:%f\n", i, (-1.)*loss/float64(i), (-1.)*lossTest/float64(len(test)))
	}

	//fmt.Printf("%v %v %v\n\n", li.LabelIdx, li.LabelVals, li.Labels)
	accuracy, cm := li.TestLinesSVM(test)
	fmt.Printf("\nFinished Testing < logistic regression mashroom >\n")
	fmt.Printf("\tAccuracy (online learn): %.2f\n\n", accuracy)
	start := time.Now()
	epoh := 0
	// warm up
	li.TrainLines(lines, epoh)
	duration := time.Now().Sub(start)
	if epoh > 0 {
		fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*epoh))
	}
	fmt.Printf("\tFutures: %d Labels: %d\n", len(li.Future), len(li.LabelIdx))

	start = time.Now()
	accuracy, cm = li.TestLinesSVM(test)
	duration = time.Now().Sub(start)
	fmt.Println("\tAverage prediction time:", duration/time.Duration(len(test)))
	fmt.Printf("\tAccuracy (offline): %.2f cm:%+v\n\n", accuracy, cm)
}

func TestIris(t *testing.T) {
	seed := 43
	filepath := "dataset/iris.csv"
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	lines := make([]string, 0, 1024)

	/*
		// svm dataset
		for scanner.Scan() {
			scanText := scanner.Text()
			lines = append(lines, scanText)
		}
	*/

	// skip header
	scanner.Scan()
	for scanner.Scan() {
		var f1, f2, f3, f4 float64
		var s string
		n, err := fmt.Sscanf(scanner.Text(), "%f,%f,%f,%f,%s", &f1, &f2, &f3, &f4, &s)
		if n != 5 || err != nil {
			continue
		}
		lines = append(lines, fmt.Sprintf("%s 1:%f 2:%f 3:%f 4:%f", s, f1, f2, f3, f4))
	}

	// shuffle
	lines = ShuffleStr(lines, seed)

	// separate training and test sets
	trainLen := int(0.9 * float64(len(lines)))
	train := lines[:trainLen]
	test := lines[trainLen:]

	fmt.Println(len(lines))
	li := flyml.LogItNew(0.001, seed)
	// online learn & load
	for _, s := range train {
		li.TrainLine(s)
		//li.LoadLineSVM(s)
	}

	accuracy, cm := li.TestLinesSVM(test)
	fmt.Printf("\nFinished Testing < logistic regression: iris >\n")
	fmt.Printf("\tAccuracy (online learn): %.2f\n\n", accuracy)
	//fmt.Printf("\t%+v\t%+v\t%v\n", li.LabelIdx, li.LabelVals, li.Labels)
	start := time.Now()
	epoh := 10000
	// warm up
	li.TrainLines(train, epoh)
	duration := time.Now().Sub(start)
	if epoh > 0 {
		fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*epoh))
	}
	fmt.Printf("\tFutures: %d Labels: %d\n", len(li.Future), len(li.LabelIdx))

	start = time.Now()
	accuracy, cm = li.TestLinesSVM(test)
	duration = time.Now().Sub(start)
	fmt.Println("\tAverage prediction time:", duration/time.Duration(len(test)))
	fmt.Printf("\tAccuracy (offline %d epoch): %.2f cm:%+v\n\n", epoh, accuracy, cm)
	//fmt.Println(test)
}

func TestBreastCancer(t *testing.T) {
	seed := 42
	filepath := "dataset/breast-cancer-scale.txt"
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	lines := make([]string, 0, 1024)

	// svm dataset
	for scanner.Scan() {
		scanText := scanner.Text()
		lines = append(lines, scanText)
	}

	// shuffle
	lines = ShuffleStr(lines, seed)

	// separate training and test sets
	trainLen := int(0.9 * float64(len(lines)))
	train := lines[:trainLen]
	test := lines[trainLen:]

	//fmt.Println(len(lines))
	li := flyml.LogItNew(0.001, seed)
	// online learn & load
	for _, s := range train {
		li.TrainLine(s)
	}

	accuracy, cm := li.TestLinesSVM(test)
	fmt.Printf("\nFinished Testing < logistic regression: breast-cancer >\n")
	fmt.Printf("\tAccuracy (online learn): %.2f\n\n", accuracy)
	//fmt.Printf("\t%+v\t%+v\t%v\n", li.LabelIdx, li.LabelVals, li.Labels)
	start := time.Now()
	epoh := 1
	// warm up
	li.TrainLines(train, epoh)
	duration := time.Now().Sub(start)
	if epoh > 0 {
		fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*epoh))
	}
	fmt.Printf("\tFutures: %d Labels: %d\n", len(li.Future), len(li.LabelIdx))

	start = time.Now()
	accuracy, cm = li.TestLinesSVM(test)
	duration = time.Now().Sub(start)
	fmt.Println("\tAverage prediction time:", duration/time.Duration(len(test)))
	fmt.Printf("\tAccuracy (offline %d epoch): %.2f cm:%+v\n\n", epoh, accuracy, cm)
	//fmt.Println(test)
}

func TestShuffle(t *testing.T) {
	str := make([]string, 0)
	for i := 0; i < 99; i++ {
		str = append(str, fmt.Sprintf("%d", i))
	}
	str = ShuffleStr(str, 42)
	ch := make(map[string]bool)
	for i := range str {
		if ok, _ := ch[str[i]]; !ok {
			ch[str[i]] = true
		} else {
			log.Fatal("err")
		}
	}
	//fmt.Printf("%+v\n", str)
}

func TestNews20(t *testing.T) {
	seed := 42
	filepath := "dataset/news20.zip"
	//unpack ria.tsv
	r, err := zip.OpenReader(filepath)
	assert.NoError(t, err)
	defer r.Close()
	for _, f := range r.File {
		dest, err := os.Create("dataset/news20")
		assert.NoError(t, err)
		defer dest.Close()
		rc, err := f.Open()
		assert.NoError(t, err)
		defer rc.Close()
		_, err = io.Copy(dest, rc)
		assert.NoError(t, err)
	}
	defer os.Remove("dataset/news20")

	filepath = "dataset/news20"
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	lines := make([]string, 0, 1024)

	// svm dataset
	for scanner.Scan() {
		scanText := scanner.Text()
		lines = append(lines, scanText)
	}

	// shuffle
	lines = ShuffleStr(lines, seed)

	// separate training and test sets
	trainLen := int(0.9 * float64(len(lines)))
	train := lines[:trainLen]
	test := lines[trainLen:]

	//fmt.Println(len(lines))
	li := flyml.LogItNew(0.001, seed)
	// online learn & load
	for _, s := range train {
		li.TrainLine(s)
	}

	accuracy, cm := li.TestLinesSVM(test)
	fmt.Printf("\nFinished Testing < logistic regression: news20 >\n")
	fmt.Printf("\tAccuracy (online learn): %.2f\n\n", accuracy)
	//fmt.Printf("\t%+v\t%+v\t%v\n", li.Label, li.LabelVals, li.Labels)
	start := time.Now()
	epoh := 500
	// warm up
	li.TrainLines(train, epoh)
	duration := time.Now().Sub(start)
	if epoh > 0 {
		fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*(epoh)))
		fmt.Println("\tTotal time:", duration)
	}
	fmt.Printf("\tFutures: %d Labels: %d\n", len(li.Future), len(li.LabelIdx))

	start = time.Now()
	accuracy, cm = li.TestLinesSVM(test)
	duration = time.Now().Sub(start)
	fmt.Println("\tAverage prediction time:", duration/time.Duration(len(test)))
	fmt.Printf("\tAccuracy (offline %d epoch): %.2f cm:%+v\n\n", epoh, accuracy, cm)
	//fmt.Println(test)
}

func TestBLAS(t *testing.T) {
	a := blas64.Vector{Inc: 1, Data: []float64{1, 2, 3}}
	b := blas64.Vector{Inc: 1, Data: []float64{2, 3, 4}}
	dot := blas64.Dot(a, b)
	fmt.Println("v dot:", dot)
}

// для весов структура
type Arm struct {
	Shows  float64
	Clicks float64
}

func TestRandom(t *testing.T) {
	// тестируем алгоритм рандом vs weited random
	// есть 2 руки с вероятностью клика 3 и 7%
	a := make(map[int]float64)
	a[0] = 0.03
	a[1] = 0.07
	steps := 100_000 // шагов в экспе
	success := 0
	success_wr := 0

	arms := make(map[int]*Arm)
	for i := 0; i < len(a); i++ {
		arms[i] = &Arm{}
	}
	rand.NewSource(int64(42)) // для предсказуемости результатов
	// перебираем
	for i := 0; i < steps; i++ {
		//выбираем рандомно руку
		variant := rand.Intn(len(a))
		monetka := rand.Float64()
		if monetka <= a[variant] {
			//кликнул ебана
			success++
		}
		// а теперь по весам выбираем при том что мы не_знаем ctr (в этом смысл этой игры)
		variant, _ = sample_pmf(arm_ctr(arms))
		arms[variant].Shows += 1.
		if monetka <= a[variant] {
			arms[variant].Clicks += 1.
			success_wr++
		}

	}
	fmt.Println("При рандоме угадали:", success, " accuracy:", float64(success)*100./float64(steps), "%")
	fmt.Println("При взв. рандоме угадали:", success_wr, " accuracy:", float64(success_wr)*100./float64(steps), "%")
	for i := range arms {
		fmt.Printf("%+v\n", arms[i])
	}
	fmt.Println(arm_ctr(arms))
	fmt.Println("Кликов больше на:", 100.-float64(success)*100./float64(success_wr), "%")

}

func arm_ctr(a map[int]*Arm) []float64 {
	ctr := make([]float64, len(a))
	for i := range a {
		if a[i].Clicks == 0 || a[i].Shows == 0 {
			ctr[i] = 1. / float64(len(a))
			continue
		}
		ctr[i] = a[i].Clicks / a[i].Shows
	}
	return ctr
}

// sample by probability mass function
func sample_pmf(pmf []float64) (int, float64) {
	var total float64 = 0.0
	for _, v := range pmf {
		total += v
	}
	scale := 1 / total

	for i, _ := range pmf {
		pmf[i] *= scale
	}
	draw := rand.Float64()
	var sum_prob float64 = 0.0
	for index, prob := range pmf {
		sum_prob += prob
		if sum_prob >= draw {
			return index, prob
		}
	}
	return 0, pmf[0]
}
