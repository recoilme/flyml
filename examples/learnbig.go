package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/recoilme/flyml"
)

func main() {
	seed := 42
	filepath := "/Users/v.kulibaba/Documents/learning_blm_cpc/model_train.tsv"
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	train := make([]string, 0, 1024)
	test := make([]string, 0, 1024)

	li := flyml.LogItNew(0.001, seed)
	// svm dataset
	i := 0
	//loss := 0.
	for scanner.Scan() {
		scanText := scanner.Text()
		if i%1000000 == 0 {
			fmt.Println(i, i/1000)
		}
		if i==1000001  {
			fmt.Println("learn",i)
			break
		}
		if i%10 == 0 {
			test = append(test, scanText)
			i++
			continue
		}
		//li.TrainLine(scanText)
		labelIdx, futures, err := li.LoadLineSVM(scanText)
		if err != nil {
			log.Fatal(err)
		}
		li.Train(futures, li.LabelVals[labelIdx], false)

		/*loss += li.Train(futures, li.LabelVals[labelIdx], true)
		lossTest := 0.
		for _, sTest := range test {
			lossTest += li.TestLineLogLoss(sTest)
		}
		fmt.Printf("ep:%d loss:%f lossTest:%f\n", i, (-1.)*loss/float64(i), (-1.)*lossTest/float64(len(test)))
		*/
		i++
		/*if i%1_000_000 == 0 {
			fmt.Printf("Mln lines:%d t:%v\n", i/1_000_000, time.Now())
			// warm up
			li.TrainLines(train, 1)
			train = make([]string, 0, 1024)
		}
		if i%10 == 0 {
			test = append(test, scanText)
		} else {
			train = append(train, scanText)
		}*/

	}

	fmt.Printf("Labels:%d Futures:%d ", len(li.Labels), len(li.Future))
	// separate training and test sets
	accuracy, cm := li.TestLinesSVM(test)
	fmt.Printf("\nFinished Testing < logistic regression: blm >\n")
	fmt.Printf("\tAccuracy (online learn): %.2f cm:%+v\n\n", accuracy, cm)
	start := time.Now()
	epoh := 0
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
	//fmt.Println(test)xs
}

func ShuffleStr(slice []string, seed int) []string {
	rand.Seed(int64(seed))
	rand.Shuffle(len(slice), func(i, j int) {
		slice[i], slice[j] = slice[j], slice[i]
	})
	return slice
}
