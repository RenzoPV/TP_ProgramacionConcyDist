package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Tipos de procedimiento
var procedimientoMap = map[string]int{
	"DOSAJE DE CREATININA EN SANGRE": 0,
	"TRIGLICERIDOS":                  1,
}

// Posibles diagnósticos
var diagnosticos = map[string]int{
	"HIPERLIPIDEMIA MIXTA":           0,
	"HIPERGLICERIDEMIA PURA":         1,
	"HIPERLIPIDEMIA NO ESPECIFICADA": 2,
	"HIPERCOLESTEROLEMIA PURA":       3,
	"TRASTORNO DEL METABOLISMO DE LAS LIPOPROTEINAS, NO ESPECIFICADO": 4,
	"OTRA HIPERLIPIDEMIA": 5,
	"OTROS TRASTORNOS DEL METABOLISMO DE LAS LIPOPROTEINAS": 6,
	"HIPERQUILOMICRONEMIA":         7,
	"DEFICIENCIA DE LIPOPROTEINAS": 8,
}

// Estructura de la Red Neuronal
type NeuralNetwork struct {
	inputNodes   int
	hiddenNodes  int
	outputNodes  int
	weightsIH    [][]float64 // Pesos de input a hidden
	weightsHO    [][]float64 // Pesos de hidden a output
	biasH        []float64   // Sesgo en la capa oculta
	biasO        []float64   // Sesgo en la capa de salida
	learningRate float64
}

// Función sigmoide para la activación
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivada de la función sigmoide
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Inicializar la red neuronal
func NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes int, learningRate float64) *NeuralNetwork {
	nn := &NeuralNetwork{
		inputNodes:   inputNodes,
		hiddenNodes:  hiddenNodes,
		outputNodes:  outputNodes,
		learningRate: learningRate,
		weightsIH:    make([][]float64, hiddenNodes),
		weightsHO:    make([][]float64, outputNodes),
		biasH:        make([]float64, hiddenNodes),
		biasO:        make([]float64, outputNodes),
	}

	// Inicializar pesos y bias con valores aleatorios
	for i := 0; i < hiddenNodes; i++ {
		nn.weightsIH[i] = make([]float64, inputNodes)
		for j := 0; j < inputNodes; j++ {
			nn.weightsIH[i][j] = rand.Float64()*2 - 1
		}
		nn.biasH[i] = rand.Float64()*2 - 1
	}

	for i := 0; i < outputNodes; i++ {
		nn.weightsHO[i] = make([]float64, hiddenNodes)
		for j := 0; j < hiddenNodes; j++ {
			nn.weightsHO[i][j] = rand.Float64()*2 - 1
		}
		nn.biasO[i] = rand.Float64()*2 - 1
	}

	return nn
}

// Función para hacer predicciones
func (nn *NeuralNetwork) predict(inputs []float64) []float64 {

	hiddenOutputs := make([]float64, nn.hiddenNodes)
	for i := 0; i < nn.hiddenNodes; i++ {
		sum := nn.biasH[i]
		for j := 0; j < nn.inputNodes; j++ {
			sum += nn.weightsIH[i][j] * inputs[j]
		}
		hiddenOutputs[i] = sigmoid(sum)
	}

	finalOutputs := make([]float64, nn.outputNodes)
	for i := 0; i < nn.outputNodes; i++ {
		sum := nn.biasO[i]
		for j := 0; j < nn.hiddenNodes; j++ {
			sum += nn.weightsHO[i][j] * hiddenOutputs[j]
		}
		finalOutputs[i] = sigmoid(sum)
	}

	return finalOutputs
}

// Retorna el índice del diagnóstico con mayor probabilidad
func (nn *NeuralNetwork) predictDiagnostico(inputs []float64) int {
	outputs := nn.predict(inputs)
	highestIdx := 0
	for i := range outputs {
		if outputs[i] > outputs[highestIdx] {
			highestIdx = i
		}
	}
	return highestIdx
}

// Cargar el dataset, procesar las características y el diagnóstico
func loadDataset(filePath string) ([][]float64, [][]float64, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';' // Separador
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	var dataset [][]float64
	var labels [][]float64

	// Procesar las filas, omitiendo la cabecera
	for _, record := range records[1:] {
		var row []float64

		// Edad del paciente
		edad, _ := strconv.ParseFloat(record[8], 64)
		row = append(row, edad)

		// Sexo - Codificación binaria
		sexo := 0.0
		if strings.ToLower(record[9]) == "masculino" {
			sexo = 1.0
		}
		row = append(row, sexo)

		// Procedimiento 1 - Label Encoding
		procedimiento1 := record[20]
		if val, ok := procedimientoMap[procedimiento1]; ok {
			row = append(row, float64(val))
		} else {
			row = append(row, 0.0) // Valor por defecto si no está en el mapa
		}

		// Resultado de Procedimiento 1
		resultado1, _ := strconv.ParseFloat(record[20], 64)
		row = append(row, resultado1)

		// Procedimiento 2 - Label Encoding
		procedimiento2 := record[23]
		if val, ok := procedimientoMap[procedimiento2]; ok {
			row = append(row, float64(val))
		} else {
			row = append(row, 0.0)
		}

		// Resultado de Procedimiento 2
		resultado2, _ := strconv.ParseFloat(record[24], 64)
		row = append(row, resultado2)

		// Añadir la fila procesada al dataset
		dataset = append(dataset, row)

		// Codificación del diagnóstico (One-Hot)
		diagnostico := make([]float64, len(diagnosticos))
		if idx, ok := diagnosticos[record[13]]; ok {
			diagnostico[idx] = 1.0
		}
		labels = append(labels, diagnostico)
	}

	return dataset, labels, nil
}

// Cálculo de precisión concurrente
func calculateAccuracyConcurrent(nn *NeuralNetwork, dataset [][]float64, labels [][]float64) float64 {
	correct := 0
	var wg sync.WaitGroup
	var mux sync.Mutex

	for i := 0; i < len(dataset); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			prediction := nn.predictDiagnostico(dataset[i])
			realLabel := 0
			for idx, val := range labels[i] {
				if val == 1.0 {
					realLabel = idx
					break
				}
			}
			if prediction == realLabel {
				mux.Lock()
				correct++
				mux.Unlock()
			}
		}(i)
	}

	wg.Wait()
	return float64(correct) / float64(len(labels)) * 100.0
}

func (nn *NeuralNetwork) trainBatchConcurrent(batch [][]float64, targetBatch [][]float64, mux *sync.Mutex, wg *sync.WaitGroup) {
	defer wg.Done()

	// Variables para acumular los gradientes
	deltaWeightsIH := make([][]float64, nn.hiddenNodes)
	deltaWeightsHO := make([][]float64, nn.outputNodes)
	deltaBiasH := make([]float64, nn.hiddenNodes)
	deltaBiasO := make([]float64, nn.outputNodes)

	// Inicializar los gradientes acumulados
	for i := 0; i < nn.hiddenNodes; i++ {
		deltaWeightsIH[i] = make([]float64, nn.inputNodes)
	}
	for i := 0; i < nn.outputNodes; i++ {
		deltaWeightsHO[i] = make([]float64, nn.hiddenNodes)
	}

	// Procesar cada instancia en el batch
	for i := 0; i < len(batch); i++ {
		inputs := batch[i]
		targets := targetBatch[i]

		hiddenOutputs := make([]float64, nn.hiddenNodes)
		for i := 0; i < nn.hiddenNodes; i++ {
			sum := nn.biasH[i]
			for j := 0; j < nn.inputNodes; j++ {
				sum += nn.weightsIH[i][j] * inputs[j]
			}
			hiddenOutputs[i] = sigmoid(sum)
		}

		finalOutputs := make([]float64, nn.outputNodes)
		for i := 0; i < nn.outputNodes; i++ {
			sum := nn.biasO[i]
			for j := 0; j < nn.hiddenNodes; j++ {
				sum += nn.weightsHO[i][j] * hiddenOutputs[j]
			}
			finalOutputs[i] = sigmoid(sum)
		}

		// Backpropagation (acumular los gradientes)
		outputErrors := make([]float64, nn.outputNodes)
		for i := 0; i < nn.outputNodes; i++ {
			outputErrors[i] = targets[i] - finalOutputs[i]
		}

		hiddenErrors := make([]float64, nn.hiddenNodes)
		for i := 0; i < nn.hiddenNodes; i++ {
			sum := 0.0
			for j := 0; j < nn.outputNodes; j++ {
				sum += nn.weightsHO[j][i] * outputErrors[j]
			}
			hiddenErrors[i] = sum
		}

		// Actualizar gradientes acumulados para capa de salida
		for i := 0; i < nn.outputNodes; i++ {
			for j := 0; j < nn.hiddenNodes; j++ {
				deltaWeightsHO[i][j] += nn.learningRate * outputErrors[i] * sigmoidDerivative(finalOutputs[i]) * hiddenOutputs[j]
			}
			deltaBiasO[i] += nn.learningRate * outputErrors[i] * sigmoidDerivative(finalOutputs[i])
		}

		// Actualizar gradientes acumulados para capa oculta
		for i := 0; i < nn.hiddenNodes; i++ {
			for j := 0; j < nn.inputNodes; j++ {
				deltaWeightsIH[i][j] += nn.learningRate * hiddenErrors[i] * sigmoidDerivative(hiddenOutputs[i]) * inputs[j]
			}
			deltaBiasH[i] += nn.learningRate * hiddenErrors[i] * sigmoidDerivative(hiddenOutputs[i])
		}
	}

	// Aplicar los gradientes acumulados
	mux.Lock()
	for i := 0; i < nn.outputNodes; i++ {
		for j := 0; j < nn.hiddenNodes; j++ {
			nn.weightsHO[i][j] += deltaWeightsHO[i][j]
		}
		nn.biasO[i] += deltaBiasO[i]
	}

	for i := 0; i < nn.hiddenNodes; i++ {
		for j := 0; j < nn.inputNodes; j++ {
			nn.weightsIH[i][j] += deltaWeightsIH[i][j]
		}
		nn.biasH[i] += deltaBiasH[i]
	}
	mux.Unlock()
}

// Función para dividir el dataset en batches
func createBatches(dataset [][]float64, targets [][]float64, batchSize int) ([][][]float64, [][][]float64) {
	var batches [][][]float64
	var targetBatches [][][]float64

	for i := 0; i < len(dataset); i += batchSize {
		end := i + batchSize
		if end > len(dataset) {
			end = len(dataset)
		}
		batches = append(batches, dataset[i:end])
		targetBatches = append(targetBatches, targets[i:end])
	}

	return batches, targetBatches
}

func ingresarDatosYPredecir(nn *NeuralNetwork) {
	reader := bufio.NewReader(os.Stdin)

	// Ingresar la edad
	fmt.Print("Ingresa la edad del paciente: ")
	edadStr, _ := reader.ReadString('\n')
	edadStr = strings.TrimSpace(edadStr)
	edad, _ := strconv.ParseFloat(edadStr, 64)

	// Ingresar el sexo
	fmt.Print("Ingresa el sexo del paciente (masculino/femenino): ")
	sexoStr, _ := reader.ReadString('\n')
	sexoStr = strings.TrimSpace(sexoStr)
	sexo := 0.0
	if strings.ToLower(sexoStr) == "masculino" {
		sexo = 1.0
	}

	// Ingresar el Procedimiento 1
	fmt.Print("Ingresa el Procedimiento 1: ")
	procedimiento1Str, _ := reader.ReadString('\n')
	procedimiento1Str = strings.TrimSpace(procedimiento1Str)
	procedimiento1 := 0.0
	if val, ok := procedimientoMap[procedimiento1Str]; ok {
		procedimiento1 = float64(val)
	}

	// Ingresar el resultado de Procedimiento 1
	fmt.Print("Ingresa el resultado de Procedimiento 1(en mg/dL): ")
	resultado1Str, _ := reader.ReadString('\n')
	resultado1Str = strings.TrimSpace(resultado1Str)
	resultado1, _ := strconv.ParseFloat(resultado1Str, 64)

	// Ingresar el Procedimiento 2
	fmt.Print("Ingresa el Procedimiento 2: ")
	procedimiento2Str, _ := reader.ReadString('\n')
	procedimiento2Str = strings.TrimSpace(procedimiento2Str)
	procedimiento2 := 0.0
	if val, ok := procedimientoMap[procedimiento2Str]; ok {
		procedimiento2 = float64(val)
	}

	// Ingresar el resultado de Procedimiento 2
	fmt.Print("Ingresa el resultado de Procedimiento 2(en mg/dL): ")
	resultado2Str, _ := reader.ReadString('\n')
	resultado2Str = strings.TrimSpace(resultado2Str)
	resultado2, _ := strconv.ParseFloat(resultado2Str, 64)

	// Crear el vector de entrada para la predicción
	inputs := []float64{edad, sexo, procedimiento1, resultado1, procedimiento2, resultado2}

	// Obtener el diagnóstico predicho
	diagnosticoIdx := nn.predictDiagnostico(inputs)

	// Buscar el diagnóstico correspondiente
	var diagnostico string
	for key, val := range diagnosticos {
		if val == diagnosticoIdx {
			diagnostico = key
			break
		}
	}

	// Mostrar el diagnóstico predicho
	fmt.Printf("El diagnóstico predicho es: %s\n", diagnostico)
}

func main() {
	start := time.Now()
	var opc int

	// Parámetros de la red neuronal
	inputNodes := 6 // Número de características del dataset
	hiddenNodes := 10
	outputNodes := 9 // Número de diagnósticos posibles
	learningRate := 0.05
	batchSize := 64

	// Crear la red neuronal
	nn := NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

	// Cargar el dataset
	trainDataset, trainLabels, err := loadDataset("data/train.csv")
	if err != nil {
		log.Fatal("Error al cargar el dataset de entrenamiento:", err)
	}
	// Definir el número de clases
	nClasses := 9

	trainTargets := make([][]float64, len(trainLabels))
	for i, label := range trainLabels {
		oneHot := make([]float64, nClasses)
		for idx, val := range label {
			if val == 1.0 {
				oneHot[idx] = 1.0 // Marcar la clase correspondiente con 1.0
			}
		}
		trainTargets[i] = oneHot
	}

	batches, targetBatches := createBatches(trainDataset, trainTargets, batchSize)

	// Entrenar la red neuronal
	epochs := 100
	var wg sync.WaitGroup
	var mux sync.Mutex
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(batches); i++ {
			wg.Add(1)
			go nn.trainBatchConcurrent(batches[i], targetBatches[i], &mux, &wg)
		}
		wg.Wait() // Esperar a que todas las goroutines terminen antes de la siguiente época
	}
	// Cargar el dataset de testeo
	testDataset, testLabels, err := loadDataset("data/test.csv")
	if err != nil {
		log.Fatal("Error al cargar el dataset de prueba:", err)
	}

	// Medir la precisión en el dataset de prueba concurrentemente
	accuracy := calculateAccuracyConcurrent(nn, testDataset, testLabels)
	fmt.Printf("Precisión del modelo: %.2f%%\n", accuracy)

	duration := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %v\n", duration)

	fmt.Println("\n============================================")
	fmt.Println("HOSPITAL SERGIO BERNALES")
	fmt.Println("============================================\n")
	for {
		fmt.Println("1) Ingresar datos para predecir")
		fmt.Println("2) Salir")
		fmt.Print("Elige una opción: ")
		fmt.Scanln(&opc)

		if opc == 1 {
			fmt.Println("\nLos procedimientos permitidos son: \n")
			fmt.Println("DOSAJE DE CREATININA EN SANGRE")
			fmt.Println("TRIGLICERIDOS\n")
			ingresarDatosYPredecir(nn)
		} else {
			fmt.Println("\nTenga un buen día")
			break
		}

	}

}
