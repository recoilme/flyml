package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"golang.org/x/text/transform"

	"github.com/recoilme/flyml"
)

var (
	// words is the main dictionary
	// we're using to base our features
	// from
	words map[string]int
	count int

	// filepaths
	amz string
	ylp string
	imd string

	sanitize transform.Transformer
)

func init() {
	words = make(map[string]int)
	count = 0

	amz = "./dataset/amazon_cells_labelled.txt"
	ylp = "./dataset/yelp_labelled.txt"
	imd = "./dataset/imdb_labelled.txt"

	sanitize = transform.RemoveFunc(func(r rune) bool {
		switch {
		case r >= 'A' && r <= 'Z':
			return false
		case r >= 'a' && r <= 'z':
			return false
		case r >= '0' && r <= '1':
			return false
		case r == ' ':
			return false
		case r == '\t':
			return false
		default:
			return true
		}
	})

	rand.Seed(42)
}

func parseLinesToData(filepath string) ([][]float64, []float64) {
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	x := [][]float64{}
	y := []float64{}

	now := time.Now()
	fmt.Printf("Starting munging from < %v > to data at %v\n", filepath, now)

	var ct int

	for scanner.Scan() {
		// remove all punctuation and convert to lower case
		scanText := scanner.Text()

		line := parseLineToText(scanText)

		text := textDecart(line[0])
		//text := strings.Split(line[0], " ")

		ct++
		if ct%500 == 0 {
			print(".")
		}

		label, err := strconv.Atoi(line[1])
		if err != nil {
			continue
		}

		l := make([]float64, len(words))

		for _, txt := range text {

			// skip small words
			if len(l) < 4 {
				continue
			}

			l[words[txt]] = 1.0
		}

		x = append(x, l)
		y = append(y, float64(label))

	}
	fmt.Printf("\nFinished munging from < %v > to data\n\tdelta: %v\n", filepath, time.Now().Sub(now))

	return x, y
}

// addWordsToGlobalMap adds words contained within
// the dataset's expected format to the global
// `words` dictionary variable
func addWordsToGlobalMap(filepath string) {
	f, err := os.Open(filepath)
	if err != nil {
		panic(err.Error())
	}
	defer f.Close()

	// create word bank
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	now := time.Now()
	fmt.Printf("Starting adding words from < %v > to word bank at %v\n", filepath, now)

	for scanner.Scan() {
		// remove all punctuation and convert to lower case
		str := parseLineToText(scanner.Text())[0]
		text := textDecart(str)
		//fmt.Println(text)

		for i := range text {

			// skip small words
			if len(text[i]) < 4 {
				continue
			}

			// add word to map if the word is present
			if _, there := words[text[i]]; !there {
				words[text[i]] = count
				count++
			}

		}
	}
	fmt.Printf("Finished adding words from < %v > to bank\n\tdelta: %v\n", filepath, time.Now().Sub(now))
}

func textDecart(s string) []string {
	words := strings.Split(s, " ")
	result := make([]string, 0, len(words)*len(words))
	result = append(result, words...)
	/*
		for i := range words {
			// skip small words
			if len(words[i]) < 0 {
				continue
			}
			for j := range words {
				if len(words[j]) < 0 {
					continue
				}
				result = append(result, words[i]+words[j])
			}
		}*/
	//fmt.Println(result)
	return result
}

// parseLineToText takes in a string, converts
// it to lowercase, removes anything not in a-z,
// and returns that, split between the sentence
// and the label
func parseLineToText(s string) []string {
	sanitized, _, _ := transform.String(sanitize, s)

	return strings.Split(strings.TrimSuffix(strings.ToLower(sanitized), "\n"), "\t")
	//fmt.Println(words)
}

func parseDataIntoPoints() ([][]float64, []float64) {
	// add words from files to dictionary
	addWordsToGlobalMap(amz)
	addWordsToGlobalMap(ylp)
	addWordsToGlobalMap(imd)

	// now go through and add each sentence to the array
	x := [][]float64{}
	y := []float64{}

	az, azY := parseLinesToData(amz)
	x = append(x, az...)
	y = append(y, azY...)

	yl, ylY := parseLinesToData(ylp)
	x = append(x, yl...)
	y = append(y, ylY...)

	im, imY := parseLinesToData(imd)
	x = append(x, im...)
	y = append(y, imY...)

	// shuffle data to mix imdb, amazon, and yelp reviews together

	now := time.Now()
	fmt.Printf("Shuffling the data now!\n\tStarted: %v\n", now)

	for i := range x {
		j := rand.Intn(i + 1)
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	}

	fmt.Printf("Finshed shuffling!\n\tTook %v\n", time.Now().Sub(now))

	return x, y
}

func abs(x float64) float64 {
	if x < 0 {
		return -1 * x
	}
	return x
}

type model interface {
	Predict([]float64, ...bool) ([]float64, error)
}

func main() {
	now := time.Now()
	fmt.Printf("Parsing Data Into Datapoints\n\tStarting at %v\n", now)

	x, y := parseDataIntoPoints()
	//fmt.Println(x[0], "\n\n\n", y[0])

	fmt.Printf("Parsing Data Finished!\n\tEnding in %v\n\t%v, %v datapoints recorded for x,y | x in R^%v\n", time.Now().Sub(now), len(x), len(y), len(words))

	// separate training and test sets
	trainLen := int(0.8 * float64(len(x)))
	train := x[:trainLen]
	trainY := y[:trainLen]

	test := x[trainLen:]
	testY := y[trainLen:]

	//LogReg.Model(train, trainY, test, testY, 200, 0.01)
	mod := &flyml.Model{Rand: rand.New(rand.NewSource(int64(42)))} //, 0.000001) // .New(0.1)
	mod.TrainSGD(train, trainY, 100)

	start := time.Now()
	accuracy := mod.Accuracy(test, testY)
	duration := time.Now().Sub(start)
	fmt.Printf("Finished Testing < logistic regression >\n\tAccuracy: %v percent\n\tExamples tested: %v\n\tAverage Classification Time: %v\n", accuracy, len(testY), duration/time.Duration(len(testY)))

}
