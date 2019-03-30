#!/bin/bash

echo "sample,bestValue,bestWeight,bestFitness,average,time" > resultados/$1Population$2M$3MaxGeneration$4.csv
for i in `seq 1 10`;
do
	go run main.go $1 $2 $3 $4 $i >> resultados/$1Population$2M$3MaxGeneration$4.csv
done 
