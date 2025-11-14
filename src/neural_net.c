struct NeuralNetwork {

}

NeuralNetwork* new_neural_network() {
	NeuralNetwork instance = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
	if (instance == NULL) return NULL;


	
	return instance;
}
