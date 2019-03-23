package agcanonico

import (
	"fmt"
	"sort"
	"time"
	"math/rand"
	"tarefa1_problema_mochila/common"
)

func isFactivelIndividual(individual uint64) bool {

	sumOfWeight := 0

	for index, weight := range common.Itens {

		if ((individual >> uint8(index)) & uint64(1)) == 1 {
			sumOfWeight = sumOfWeight + weight[0]
		}
	}

	return sumOfWeight <= common.LimitOfWeight && sumOfWeight > 0
}

func getFitness(individual uint64) int {

	fitness := 0

	for index, value := range common.Itens {

		if ((individual >> uint8(index)) & uint64(1)) == 1 {
			fitness = fitness + value[1]
		}
	}

	return fitness
}

func getFitnessOfPopulation(population []uint64) []int {

	var fitnessOfPopulation []int

	for _, individual := range population {

		fitnessOfPopulation = append(fitnessOfPopulation, getFitness(individual))
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

func getFactivelIndividual() uint64 {

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

		if isFactivelIndividual(individual) {
			return individual
		}
	}	
}

func getInitialPopulation(sizeOfPopulation int) []uint64 {

	var population []uint64

	for {

		population = append(population, getFactivelIndividual())

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

func rouletteOfPopulation(population []uint64) []uint64 {

	var selected []uint64

	fitnessOfPopulation := getFitnessOfPopulation(population)

	normalizedFitnessOfPopulation := getNormalizedFitnessOfPopulation(fitnessOfPopulation)

	index := 0

	for {

		selected = append(selected, roulette(normalizedFitnessOfPopulation, population))

		index++

		if index == common.SizeOfPopulation {
			break
		}
	}

	return selected
}

func crossover(parent1 uint64, parent2 uint64) []uint64 {

	for {
		rand.Seed(time.Now().UnixNano())

		cut := uint8(rand.Intn(42) + 1)
	
		mask := ((^uint64(0)) >> cut) << cut
		nmask := (^uint64(0)) ^ mask

		child1 := (mask & parent1) | (nmask & parent2)
		child2 := (mask & parent2) | (nmask & parent1)
	
		if isFactivelIndividual(child1) && isFactivelIndividual(child2) {
			return []uint64{ child1, child2 }
		}
	}
}

func crossoverOfPopulation(selectedPopulation []uint64) []uint64 {

	var childPopulation []uint64

	index := 0

	for {

		childPopulation = append(childPopulation, crossover(selectedPopulation[index], selectedPopulation[index + 1])...)

		index += 2

		if common.SizeOfPopulation == len(childPopulation) {
			break
		}
	}

	return childPopulation
}

func mutate(child uint64) uint64 {

	return child
}

func mutationOfPopulation(childPopulation []uint64) []uint64 {

	var mutatePopulation []uint64

	for _, child := range childPopulation {

		if mutatePopulation == nil {
			mutatePopulation = []uint64{ mutate(child) }
			continue
		}

		mutatePopulation = append(mutatePopulation, mutate(child))
	}

	return mutatePopulation
}

type individual struct {
	chromosome uint64
	fitness int 
}

type byFitness []individual

func (a byFitness) Len() int           { return len(a) }
func (a byFitness) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byFitness) Less(i, j int) bool { return a[i].fitness < a[j].fitness }

func selectNewPopulation(population []uint64, childPopulation []uint64) []uint64 {

	var newPopulation []individual

	mergePopulation := append(population, childPopulation...)

	fitnessOfPopulation := getFitnessOfPopulation(mergePopulation)

	for index, fitness := range fitnessOfPopulation {

		newPopulation = append(newPopulation, individual{chromosome: mergePopulation[index], fitness: fitness})
	}

	sort.Sort(byFitness(newPopulation))

	var newPopulationR []uint64
	index := common.SizeOfPopulation

	for {

		newPopulationR = append(newPopulationR, newPopulation[index].chromosome)

		index++

		if index == 2*common.SizeOfPopulation {
			break
		}
	}

	return newPopulationR
}

//Run - Executa algorítmo genético canónico
func Run() {

	population := getInitialPopulation(common.SizeOfPopulation)

	fmt.Println(getMaxFitnessOfPopulation(getFitnessOfPopulation(population)), getAverageFitnessOfPopulation(getFitnessOfPopulation(population)))
	
	generation := 0
	for {

		selected := rouletteOfPopulation(population)

		childPopulation := crossoverOfPopulation(selected)

		mutatePopulation := mutationOfPopulation(childPopulation)

		population = selectNewPopulation(population, mutatePopulation)

		fmt.Println(getMaxFitnessOfPopulation(getFitnessOfPopulation(population)), getAverageFitnessOfPopulation(getFitnessOfPopulation(population)))
	
		generation++

		if generation == common.MaxGeneration {
			break;
		}
	}
}