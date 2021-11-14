package flyml_test

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/recoilme/flyml"
)

/*func TestReg(t *testing.T) {
	reg := &flyml.Model{Rand: rand.New(rand.NewSource(int64(42)))}

	x, y := svmRead("dataset/mushrooms.svm")
	// separate training and test sets
	trainLen := int(0.8 * float64(len(x)))
	train := x[:trainLen]
	trainY := y[:trainLen]

	test := x[trainLen:]
	testY := y[trainLen:]
	start := time.Now()
	epoh := 300
	reg.TrainSGD(train, trainY, epoh)
	duration := time.Now().Sub(start)
	fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*epoh))
	start = time.Now()
	accuracy := reg.Accuracy(test, testY)
	duration = time.Now().Sub(start)
	fmt.Printf("Finished Testing < logistic regression >\n\tAccuracy: %v percent\n\tExamples tested: %v\n\tAverage Classification Time: %v\n", accuracy, len(testY), duration/time.Duration(len(testY)))
}*/

func TestReg2(t *testing.T) {
	//reg := &flyml.Model{Rand: rand.New(rand.NewSource(int64(42)))}

	x, y := svmRead("dataset/mushrooms.svm")
	// separate training and test sets
	trainLen := int(0.8 * float64(len(x)))
	train := x[:trainLen]
	trainY := y[:trainLen]

	test := x[trainLen:]
	testY := y[trainLen:]
	start := time.Now()
	epoh := 40
	//reg.TrainSGD(train, trainY, epoh)
	w := flyml.LogisticRegression(train, trainY, .1, epoh)
	duration := time.Now().Sub(start)
	fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*epoh))
	start = time.Now()
	accuracy := flyml.Accuracy(test, testY, w.RawVector().Data)
	duration = time.Now().Sub(start)
	fmt.Printf("Finished Testing < logistic regression >\n\tAccuracy: %v percent\n\tExamples tested: %v\n\tAverage Classification Time: %v\n", accuracy, len(testY), duration/time.Duration(len(testY)))
	//fmt.Println(testY)
}

func svmRead(filepath string) (x [][]float64, y []float64) {
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	// create word bank
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	fmt.Printf("Starting adding futures from < %v >\n", filepath)
	fut := make(map[int]int, 0)
	count := 0
	for scanner.Scan() {
		scanText := scanner.Text()
		fields := strings.Fields(scanText)
		for i := range fields {
			if i == 0 {
				continue //label
			}
			f := strings.TrimSuffix(fields[i], ":1")
			fidx, err := strconv.Atoi(f)
			if err != nil {
				continue
			}
			if _, ok := fut[fidx]; !ok {
				fut[fidx] = count
				count++
			}
		}
	}
	fmt.Println("\tFutures:", len(fut))

	f2, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f2.Close()

	scanner = bufio.NewScanner(f2)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		scanText := scanner.Text()
		fields := strings.Fields(scanText)
		l := make([]float64, len(fut))
		for i := range fields {
			if i == 0 {
				label, err := strconv.Atoi(fields[i])
				if err != nil {
					break
				}
				if label != 1 {
					label = 0
				}
				y = append(y, float64(label))
				continue
			}
			f := strings.TrimSuffix(fields[i], ":1")
			fidx, err := strconv.Atoi(f)
			if err != nil {
				break
			}
			l[fut[fidx]] = 1.0
		}
		x = append(x, l)
	}
	//shuffle
	for i := range x {
		j := rand.Intn(i + 1)
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	}
	return x, y
}

func TestSvmRead(t *testing.T) {
	svmRead("dataset/mushrooms.svm")
}
