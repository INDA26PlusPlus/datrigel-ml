use std::{ops::Add, ptr::addr_eq};

// 28 x 28 pixlar per bild
use mnist::*;
use matrixmultiply::*;


fn main() {
    println!("Hello, world!");
}

struct Neuron {
    weights: Vec<f64>,
    bias: f64
}

struct Network {
    input_layer: [f64; 784], // kanske inte behövs, kan nog abstraheras bort -> input direkt in i 
    hidden_layer_one: [Neuron; 20],
    // hidden_layer_two: [Neuron; 20],
    output_layer: [Neuron; 10]
}

impl Network {
    fn init_network(self) {

    }
    
    fn backpropagation(self) {

    }

    fn weight_matrix(self, layer: i8) {

    }

    // Feedforward for a specific neuron
    fn feedforward(&mut self, inputs_list: &Vec<f64>) -> Vec<f64> {
        let mut output_list: Vec<f64> = Vec::new();
        for neuron in &self.hidden_layer_one {
            output_list.push(sum_weights(inputs_list, &neuron.weights, &neuron.bias));
        }
        output_list
    }
}

impl Neuron {
    // Takes in a vec of all previous nurons activations, then applies weights and outputs its own fiering.
    fn activation(&mut self, input: Vec<f64>) -> f64{
        // psudo kåd
        let sum = sum_weights(&input, &self.weights, &self.bias);
        let output = sigmoid(sum);

        output
    }
}

// performs sigmoid floid on input f64.
fn sigmoid(input: f64) -> f64 {
    let e = std::f64::consts::E;
    let x = 1.0 / ( e.powf(-input).add(1.0));
    x
}
// returns the value of the sigmoid derivative at input.
fn sigmoid_derivative(input: f64) -> f64 {
    input * (1.0 - input)
}

// Matrix representation

// Takes in the Neurons weight list and sums all fiering.
fn sum_weights(inputs_list: &Vec<f64>, weights: &Vec<f64>, bias: &f64) -> f64 {
    // Pairs two items together and then multiplys them.
    let weighted_list = inputs_list.iter().zip(weights).map(|(x,y)| x * y);
    // sums all elements together
    let weight_sum: f64 = weighted_list.sum();

    weight_sum + bias
}

// takes in the output from the network and then calculates cost.
fn cost_function(output: [f64; 10], correct: [f64; 10]) {
    let cost = output.iter().zip(correct).map(|(x,y)|(x-y).powi(2));
}
