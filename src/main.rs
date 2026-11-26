use std::{ops::Add, ptr::addr_eq};

// 28 x 28 pixlar per bild
use mnist::*;
use ndarray::Array2;
use rand::{Rng, rng};

use crate::layer::{hidden_layer_one, hidden_layer_two, output_layer};


fn main() {
    let mut network = Network::init_network();
    
    let mut input = vec![0.0; 784];
    
    for i in 0..=460 {
        input[i] = 1.0;
    }

    let hidden_output = network.feedforward(&input, hidden_layer_one);
    //println!("{:?}", hidden_output);

    let output = network.feedforward(&hidden_output, output_layer);

    //print!("{:?}", output)
    let mut correct = vec![0.0;10];
    correct[0] = 1.0;

    print!("{:?}", cost_function(output, correct));


    
}

enum layer {
    hidden_layer_one,
    hidden_layer_two,
    output_layer
}

#[derive(Clone)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64
}

struct Network {
    input_layer: [f64; 784], // kanske inte behövs, kan nog abstraheras bort -> input direkt in i 
    hidden_layer_one: Vec<Neuron>,
    // hidden_layer_two: [Neuron; 20],
    output_layer: Vec<Neuron>
}

impl Network {
    fn init_network() -> Network {
        let mut hidden_layer_neuron_list: Vec<Neuron> = Vec::new();
        for _i in 0..20 {
            hidden_layer_neuron_list.push(Neuron::init_neuron(784));
        };

        let mut output_layer_neuron_list: Vec<Neuron> = Vec::new();
        for _i in 0..10 {
            output_layer_neuron_list.push(Neuron::init_neuron(20));
        };

        Network { 
            input_layer: ([0.0; 784]), 
            hidden_layer_one: (hidden_layer_neuron_list), 
            output_layer: (output_layer_neuron_list) 
        }
    }
    
    fn train(&mut self) {
        
    }

    fn weight_matrix(self, layer: i8) {

    }

    // Feedforward for a specific layer. Creates a vec where the first element is the first neuron...
    fn feedforward(&mut self, inputs_list: &Vec<f64>, select_layer: layer) -> Vec<f64> {
        let mut output_list: Vec<f64> = Vec::new();
        let self_layer = match select_layer {
            hidden_layer_one => &self.hidden_layer_one,
            hidden_layer_two => todo!("No hidden layer two"),
            output_layer => &self.output_layer
        };
        for neuron in self_layer {
            output_list.push(sigmoid(sum_weights(inputs_list, &neuron.weights, &neuron.bias)));
        }
        output_list
    }
}  

impl Neuron {
    // Inits one neuron with a input argument to specify how many connections are from the previous layer.
    fn init_neuron(input_connections: usize) -> Neuron {
        let mut weight_list: Vec<f64> = Vec::new();
        for i in 0..input_connections {
            weight_list.push(rand::random_range(-5.0..=5.0));
        };
        let bias: f64 = 0.0;

        let new_neuron = Neuron { weights: (weight_list), bias: (bias) };
        new_neuron
    }

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
fn cost_function(output: Vec<f64>, correct: Vec<f64>) -> Vec<f64>  {
    let cost = output.iter().zip(correct).map(|(x,y)|(x-y).powi(2));
    cost.collect()
}
