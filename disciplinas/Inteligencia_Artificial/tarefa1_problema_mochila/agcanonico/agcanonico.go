package agcanonico

import (
	"fmt"
	"sort"
	"time"
	"math/rand"
	"strconv"
	"tarefa1_problema_mochila/common"
)

func isFactivelIndividual(individual uint64) bool {

	sumOfWeight := getMaxWeight(individual)

	return sumOfWeight <= common.LimitOfWeight && sumOfWeight > 0
}

func getMaxWeight(individual uint64) int {

	sumOfWeight := 0

	for index, weight := range common.Itens {

		if ((individual >> uint8(index)) & uint64(1)) == 1 {
			sumOfWeight = sumOfWeight + weight[0]
		}
	}

	return sumOfWeight
}

func getFitness(individual uint64, penalize bool) int {

	fitness := 0
	penalizeFitness := 0
	maxWeight := 0

	for index, value := range common.Itens {

		if ((individual >> uint8(index)) & uint64(1)) == 1 {
			fitness = fitness + value[1]
			maxWeight = maxWeight + value[0]
			if maxWeight <= common.LimitOfWeight {
				penalizeFitness = penalizeFitness + value[1]
			}
		}
	}

	if penalize && !isFactivelIndividual(individual) {
		fitness = penalizeFitness		
	}

	return fitness
}

func getFitnessOfPopulation(population []uint64, penalize bool) []int {

	var fitnessOfPopulation []int

	for _, individual := range population {

		fitnessOfPopulation = append(fitnessOfPopulation, getFitness(individual, penalize))
	}

	return fitnessOfPopulation
}

func getSumFitnessOfPopulation(fitnessOfPopulation []int) int {

	sum := 0

	for _, fitness := range fitnessOfPopulation {
		sum = sum + fitness
	}

	return sum
}

func getMaxFitnessOfPopulation(fitnessOfPopulation []int) int {

	max := 0

	for _, fitness := range fitnessOfPopulation {
		if max < fitness {
			max = fitness
		}
	}

	return max
}

func getBestIndividual(population []uint64, fitnessOfPopulation []int) uint64 {

	max := 0
	index := 0

	for i, fitness := range fitnessOfPopulation {
		if max < fitness {
			max = fitness
			index = i
		}
	}

	return population[index]
}

func getAverageFitnessOfPopulation(fitnessOfPopulation []int) int {

	return getSumFitnessOfPopulation(fitnessOfPopulation)/len(fitnessOfPopulation)
}

func getNormalizedFitnessOfPopulation(fitnessOfPopulation []int) []float32 {

	var normalizedFitnessOfPopulation []float32

	sum := getSumFitnessOfPopulation(fitnessOfPopulation)

	for _, fitness := range fitnessOfPopulation {
		normalizedFitnessOfPopulation = append(normalizedFitnessOfPopulation, float32(fitness)/float32(sum))
	}

	return normalizedFitnessOfPopulation
}

func getFactivelIndividual(repair bool) uint64 {

	for {
		individual := uint64(0)

		index := uint8(0)
		for {
			rand.Seed(time.Now().UnixNano())

			if rand.Int31() & 0x0001 == 0 {
				individual = individual | (1 << index)
			}

			if index == uint8(len(common.Itens) - 1) {
				break
			}

			index++
		}

		if !repair {
			return individual
		}

		if isFactivelIndividual(individual) {
			return individual
		}
	}	
}

func getInitialPopulation(sizeOfPopulation int, repair bool) []uint64 {

	var population []uint64

	for {

		population = append(population, getFactivelIndividual(repair))

		if len(population) == sizeOfPopulation {
			break;
		}
	}

	return population
}

func roulette(normalizedFitnessOfPopulation []float32, population []uint64) uint64 {

	rand.Seed(time.Now().UnixNano())

	r := rand.Float32()
	index := 0
	sum := normalizedFitnessOfPopulation[index]

	for {

		if sum > r {
			break;
		}

		index++
		sum = sum + normalizedFitnessOfPopulation[index]
	}

	return population[index]
}

func rouletteOfPopulation(population []uint64, penalize bool) []uint64 {

	var selected []uint64

	fitnessOfPopulation := getFitnessOfPopulation(population, penalize)

	normalizedFitnessOfPopulation := getNormalizedFitnessOfPopulation(fitnessOfPopulation)

	index := 0

	for {

		selected = append(selected, roulette(normalizedFitnessOfPopulation, population))

		index++

		if index == common.M*2 {
			break
		}
	}

	return selected
}

func crossover(parent1 uint64, parent2 uint64) []uint64 {

	rand.Seed(time.Now().UnixNano())

	if rand.Intn(100) + 1 < common.Pc {
		return []uint64{parent1, parent2}
	}

	cut := uint8(rand.Intn(42) + 1)

	mask := ((^uint64(0)) >> cut) << cut
	nmask := (^uint64(0)) ^ mask

	child1 := (mask & parent1) | (nmask & parent2)
	child2 := (mask & parent2) | (nmask & parent1)

	return []uint64{ child1, child2 }		
}

func mutate(child uint64) uint64 {

	mutateChild := child

	index := uint8(0)
	for {
		if index == 42 {
			break;
		}

		rand.Seed(time.Now().UnixNano())

		if rand.Intn(100) + 1 < common.Pm {
			mutateChild = mutateChild ^ (uint64(1) << index)
		}

		index++
	}

	return mutateChild
}

func repair(child uint64) uint64 {

	if isFactivelIndividual(child) {
		return child
	}

	for {
		rand.Seed(time.Now().UnixNano())

		aux := child
		index := uint8(0)
		var indexes []uint8
		for {
			if (aux & uint64(1)) == 1 {
				indexes = append(indexes, index)
			}
			aux = aux >> 1
			index++

			if index == uint8(len(common.Itens) - 1) {
				break
			}
		}

		child = child & ^(1 << indexes[rand.Intn(len(indexes))])

		if isFactivelIndividual(child) {
			break
		}

		index++
	}

	return child
}

func crossoverAndMutateOfPopulation(selectedPopulation []uint64, repar bool) []uint64 {

	var childPopulation []uint64

	index := 0

	for {

		childs := crossover(selectedPopulation[index], selectedPopulation[index + 1]);

		childs = []uint64{mutate(childs[0]), mutate(childs[1])}

		if repar {
			childs = []uint64{repair(childs[0]), repair(childs[1])}
		}

		childPopulation = append(childPopulation, childs...)

		index += 2

		if common.M*2 == len(childPopulation) {
			break
		}
	}

	return childPopulation
}

type individual struct {
	chromosome uint64
	fitness int 
}

type byFitness []individual

func (a byFitness) Len() int           { return len(a) }
func (a byFitness) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byFitness) Less(i, j int) bool { return a[i].fitness < a[j].fitness }

func selectNewPopulation(population []uint64, childPopulation []uint64, penalize bool) []uint64 {

	var newPopulation []individual

	mergePopulation := append(population, childPopulation...)

	fitnessOfPopulation := getFitnessOfPopulation(mergePopulation, penalize)

	for index, fitness := range fitnessOfPopulation {

		newPopulation = append(newPopulation, individual{chromosome: mergePopulation[index], fitness: fitness})
	}

	sort.Sort(byFitness(newPopulation))

	var newPopulationR []uint64
	index :=  2*common.M

	for {

		newPopulationR = append(newPopulationR, newPopulation[index].chromosome)

		index++

		if index == len(newPopulation) {
			break
		}
	}

	return newPopulationR
}

//Run - Executa algorítmo genético canónico
func Run(repar bool) {

	population := getInitialPopulation(common.SizeOfPopulation, repar)

	bestIndividual := getBestIndividual(population, getFitnessOfPopulation(population, !repar))

	fmt.Println(bestIndividual, getMaxWeight(bestIndividual), getMaxFitnessOfPopulation(getFitnessOfPopulation(population, !repar)), getAverageFitnessOfPopulation(getFitnessOfPopulation(population, !repar)))
	
	generation := 0
	for {

		selected := rouletteOfPopulation(population, !repar)

		newPopulation := crossoverAndMutateOfPopulation(selected, repar)

		population = selectNewPopulation(population, newPopulation, !repar)

		bestIndividual := getBestIndividual(population, getFitnessOfPopulation(population, !repar))

		fmt.Println(bestIndividual, getMaxWeight(bestIndividual), getMaxFitnessOfPopulation(getFitnessOfPopulation(population, !repar)), getAverageFitnessOfPopulation(getFitnessOfPopulation(population, !repar)))
	
		generation++

		if generation == common.MaxGeneration {
			break;
		}
	}

	fmt.Println("Melhor individio:")
	for index := range common.Itens {

		if ((bestIndividual >> uint8(index)) & uint64(1)) == 1 {
			fmt.Println("Item: " + strconv.Itoa((index + 1)))
		}
	}
}