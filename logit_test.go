package flyml_test

import (
	"bufio"
	"fmt"
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
	// separate training and test sets
	trainLen := int(0.9 * float64(len(lines)))
	train := lines[:trainLen]

	li := flyml.LogItNew(0.1)
	// online learn & load
	for _, s := range train {
		li.TrainLineSVM(s)
	}
	test := lines[trainLen:]
	accuracy := li.TestLinesSVM(test)
	fmt.Printf("\nFinished Testing < logistic regression >\n")
	fmt.Printf("\tAccuracy (online learn/1 epoch): %.2f\n\n", accuracy)
	start := time.Now()
	epoh := 3
	// warm up with 3 epochs
	li.TrainLinesSVM(lines, epoh)
	duration := time.Now().Sub(start)
	fmt.Println("\tAverage iter time:", duration/time.Duration(len(train)*epoh))
	fmt.Printf("\tFutures: %d Labels: %d\n", len(li.Future), len(li.Label))

	start = time.Now()
	accuracy = li.TestLinesSVM(test)
	duration = time.Now().Sub(start)
	fmt.Println("\tAverage prediction time:", duration/time.Duration(len(test)))
	fmt.Printf("\tAccuracy (offline/3 epoch): %.2f\n\n", accuracy)
}
