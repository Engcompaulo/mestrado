package main

import (
	"os"
	"fmt"
	"strconv"
)

func main() {

	nome_conjunto_fuzzy := os.Args[2]
	altura, _ := strconv.ParseFloat(os.Args[3], 32)
	step, _ := strconv.ParseFloat(os.Args[4], 32)
	alpha_corte, _ := strconv.ParseFloat(os.Args[5], 32)

	funcao_triangular_baixo := [3]float64{1, 1, 1.5}
	funcao_triangular_medio := [3]float64{1, 1.5, 2}
	funcao_triangular_alto := [3]float64{1.65, 2, 2}

	funcao_trapezoidal_baixo := [4]float64{1, 1, 0.7, 1.5}
	funcao_trapezoidal_medio := [4]float64{1, 1.4, 1.6, 2}
	funcao_trapezoidal_alto := [4]float64{1.5, 1.8, 2, 2}

	var funcao_triagular_escolhida [3]float64
	var funcao_trapezoidal_escolhida [4]float64

	if nome_conjunto_fuzzy == "baixo" {
		funcao_triagular_escolhida = funcao_triangular_baixo
		funcao_trapezoidal_escolhida = funcao_trapezoidal_baixo
	} else if nome_conjunto_fuzzy == "medio" {
		funcao_triagular_escolhida = funcao_triangular_medio
		funcao_trapezoidal_escolhida = funcao_trapezoidal_medio
	} else {
		funcao_triagular_escolhida = funcao_triangular_alto
		funcao_trapezoidal_escolhida = funcao_trapezoidal_alto
	}

	print_grau_pertinencia_funcao_triangular(funcao_triagular_escolhida, funcao_triagular_escolhida[1], altura)
	print_grau_pertinencia_funcao_triangular(funcao_triagular_escolhida, step, altura)
	fmt.Printf("Suporte Escolhido: %f < x < %f\n", funcao_triagular_escolhida[0], funcao_triagular_escolhida[2])
	fmt.Printf("Nucleo Escolhido: %f\n", funcao_triagular_escolhida[1])
	fmt.Printf("Altura: 1\n")
	fmt.Printf("Alpha Corte: %f < x < %f\n", (funcao_triagular_escolhida[1] - funcao_triagular_escolhida[0])*alpha_corte + funcao_triagular_escolhida[0], funcao_triagular_escolhida[2] - (funcao_triagular_escolhida[2] - funcao_triagular_escolhida[1])*alpha_corte)
	

	print_grau_pertinencia_funcao_trapezoidal(funcao_trapezoidal_escolhida, funcao_trapezoidal_escolhida[1], altura)
	print_grau_pertinencia_funcao_trapezoidal(funcao_trapezoidal_escolhida, step, altura)
	fmt.Printf("Suporte Escolhido: %f < x < %f\n", funcao_trapezoidal_escolhida[0], funcao_trapezoidal_escolhida[3])
	fmt.Printf("Nucleo Escolhido: %f < x < %f\n", funcao_trapezoidal_escolhida[1], funcao_trapezoidal_escolhida[2])
	fmt.Printf("Altura: 1\n")
	fmt.Printf("Alpha Corte: %f < x < %f\n", (funcao_trapezoidal_escolhida[1] - funcao_trapezoidal_escolhida[0])*alpha_corte + funcao_trapezoidal_escolhida[0], funcao_trapezoidal_escolhida[3] - (funcao_trapezoidal_escolhida[3] - funcao_trapezoidal_escolhida[2])*alpha_corte)
	
}

func print_grau_pertinencia_funcao_triangular(funcao_triagular [3]float64, step float64, altura float64) {
	if altura <= funcao_triagular[0] {
		fmt.Printf("Grau de pertinencia = 0\n")
	} else if altura > funcao_triagular[0] && altura <= step {
		fmt.Printf("Grau de pertinencia = %f\n", (altura - funcao_triagular[0])/(step - funcao_triagular[0]))
	} else if altura > step && altura <= funcao_triagular[2] {
		fmt.Printf("Grau de pertinencia = %f\n", (funcao_triagular[2] - altura)/(funcao_triagular[2] - step))
	} else if altura > funcao_triagular[2] {
		fmt.Printf("Grau de pertinencia = 0\n")
	}
}

func print_grau_pertinencia_funcao_trapezoidal(funcao_trapezoidal [4]float64, step float64, altura float64) {
	if altura <= funcao_trapezoidal[0] {
		fmt.Printf("Grau de pertinencia = 0\n")
	} else if altura > funcao_trapezoidal[0] && altura <= step {
		fmt.Printf("Grau de pertinencia = %f\n", (altura - funcao_trapezoidal[0])/(step - funcao_trapezoidal[0]))
	} else if altura > step && altura <= funcao_trapezoidal[2] {
		fmt.Printf("Grau de pertinencia = 1\n")
	} else if altura > funcao_trapezoidal[2] && altura <= funcao_trapezoidal[3] {
		fmt.Printf("Grau de pertinencia = %f\n", (funcao_trapezoidal[3] - altura)/(funcao_trapezoidal[3] - funcao_trapezoidal[2]))
	} else if altura > funcao_trapezoidal[3] {
		fmt.Printf("Grau de pertinencia = 0\n")
	}
}