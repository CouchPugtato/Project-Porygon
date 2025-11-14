#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <stddef.h>

typedef struct NeuralNetwork NeuralNetwork;

NeuralNetwork* nn_create(int input_dim, const int* hidden_dims, int hidden_count, int num_actions);

void nn_destroy(NeuralNetwork* nn);

// forward: policy (softmax) and value
void nn_forward(NeuralNetwork* nn, const double* input, double* policy_out, double* value_out);

// backward: apply policy/value gradients
void nn_backward(NeuralNetwork* nn, const double* input, int action_index, double advantage, double target_value);

// apply SGD update
void nn_update(NeuralNetwork* nn, double learning_rate);

// one hot utility function
void one_hot(int index, int length, double* out);

#endif