package flyml_test

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/recoilme/flyml"
	"github.com/stretchr/testify/assert"
)

func TestSVM(t *testing.T) {
	li := flyml.LogItNew(0.1)
	_, _, err := li.LoadLineSVM("1 6:1 8:1 15:1 21:1 29:1 33:1 34:1 37:1 42:1 50:1")
	assert.NoError(t, err)
	_, fut, err := li.LoadLineSVM("2 7:1 8:1")
	assert.NoError(t, err)
	fmt.Printf("%+v\n", li)
	// &{Label:map[1:0 2:1]
	// Future:map[6:0 7:10 8:1 15:2 21:3 29:4 33:5 34:6 37:7 42:8 50:9]}
	// [0 1 0 0 0 0 0 0 0 0 1]
	assert.Equal(t, 11, len(fut))
	assert.Equal(t, 2, len(li.Label))
}

func TestMashroom(t *testing.T) {
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
	rand.NewSource(int64(42))
	rand.Shuffle(len(lines), func(i, j int) {
		lines[i], lines[j] = lines[j], lines[i]
	})

	// separate training and test sets
	trainLen := int(0.9 * float64(len(lines)))
	train := lines[:trainLen]

	li := flyml.LogItNew(0.001)
	// online learn & load
	for _, s := range train {
		li.TrainLine(s)
	}

	fmt.Printf("%v %v %v\n\n", li.Label, li.LabelVals, li.Labels)
	test := lines[trainLen:]
	accuracy := li.TestLinesSVM(test)
	fmt.Printf("\nFinished Testing < logistic regression mashroom >\n")
	fmt.Printf("\tAccuracy (online learn): %.2f\n\n", accuracy)
	start := time.Now()
	epoh := 51
	// warm up
	li.TrainLines(lines, epoh)
	duration := time.Now().Sub(start)
	if epoh > 0 {
		fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*epoh))
	}
	fmt.Printf("\tFutures: %d Labels: %d\n", len(li.Future), len(li.Label))

	start = time.Now()
	accuracy = li.TestLinesSVM(test)
	duration = time.Now().Sub(start)
	fmt.Println("\tAverage prediction time:", duration/time.Duration(len(test)))
	fmt.Printf("\tAccuracy (offline): %.2f\n\n", accuracy)
}

func TestIris(t *testing.T) {
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
		lines = append(lines, fmt.Sprintf("%s 1:%f 2:%f", s, f1, f2))
	}

	// shuffle
	rand.NewSource(int64(time.Now().Nanosecond()))
	rand.Shuffle(len(lines), func(i, j int) {
		lines[i], lines[j] = lines[j], lines[i]
	})

	// separate training and test sets
	trainLen := int(0.75 * float64(len(lines)))
	train := lines[:trainLen]
	test := lines[trainLen:]

	fmt.Println(len(lines))
	li := flyml.LogItNew(0.001)
	// online learn & load
	for _, s := range train {
		li.TrainLine(s)
		//li.LoadLineSVM(s)
	}

	accuracy := li.TestLinesSVM(test)
	fmt.Printf("\nFinished Testing < logistic regression: iris >\n")
	fmt.Printf("\tAccuracy (online learn/1 epoch): %.2f\n\n", accuracy)
	fmt.Printf("\t%+v\t%+v\t%v\n", li.Label, li.LabelVals, li.Labels)
	start := time.Now()
	epoh := 5001
	// warm up
	li.TrainLines(train, epoh)
	duration := time.Now().Sub(start)
	if epoh > 0 {
		fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*epoh))
	}
	fmt.Printf("\tFutures: %d Labels: %d\n", len(li.Future), len(li.Label))

	start = time.Now()
	accuracy = li.TestLinesSVM(test)
	duration = time.Now().Sub(start)
	fmt.Println("\tAverage prediction time:", duration/time.Duration(len(test)))
	fmt.Printf("\tAccuracy (offline %d epoch): %.2f\n\n", epoh, accuracy)
	//fmt.Println(test)
}
