package main

import (
	"os"
	"tarefa1_problema_mochila/agcanonico"
	"strconv"
)

func main() {

	arg2, _ := strconv.Atoi(os.Args[2])
	arg3, _ := strconv.Atoi(os.Args[3])
	arg4, _ := strconv.Atoi(os.Args[4])
	arg5, _ := strconv.Atoi(os.Args[5])

	agcanonico.Run(os.Args[1] != "Penalize", arg2, arg3, arg4, arg5)	
}