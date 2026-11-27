use std::{ops::{Add, Index}, ptr::addr_eq};

use mnist::MnistBuilder;
// 28 x 28 pixlar per bild
use ndarray::{Array, Array2, ArrayView, s};
use rand::{Rng, rng};

use crate::layer::{hidden_layer_one, hidden_layer_two, output_layer};


fn main() {
    let mut network = Network::init_network();
    let mnist = MnistBuilder::new()
        .base_url("https://raw.githubusercontent.com/fgnt/mnist/master/")
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .download_and_extract()
        .finalize();

    let mut generation_counter = 0;
    loop {
    let mut image_index = 0;
    let mut batch_buff: Vec<Vec<f64>> = Vec::new();
    let mut label_buff: Vec<Vec<f64>> = Vec::new();

for i in 0..10 {
    // fetch batch
    for image_index in image_index..(32 + image_index) {
        let start = image_index * 784;
        let end = start + 784;
        let image = mnist.trn_img[start..end].to_vec();
        let label = mnist.trn_lbl[image_index];
        let fromated_image: Vec<f64> = image.iter().map(|&pixel| pixel as f64 / 255.0).collect();

        batch_buff.push(fromated_image);
        let mut zero_vec: Vec<f64> = vec![0.0; 10];
        zero_vec[label as usize] = 1.0;
        label_buff.push(zero_vec);
    }
    network.train(&batch_buff, &label_buff);
}
    network.learning_rate = network.learning_rate / 1.5;

    let mut image_index_test = 0;
    let mut batch_buff_test: Vec<Vec<f64>> = Vec::new();
    let mut label_buff_test: Vec<Vec<f64>> = Vec::new();

    // fetch batch
    for image_index_test in image_index_test..1000 {
        let start = image_index_test * 784;
        let end = start + 784;
        let image = mnist.tst_img[start..end].to_vec();
        let label = mnist.tst_lbl[image_index_test];
        let fromated_image: Vec<f64> = image.iter().map(|&pixel| pixel as f64 / 255.0).collect();

        batch_buff_test.push(fromated_image);
        let mut zero_vec: Vec<f64> = vec![0.0; 10];
        zero_vec[label as usize] = 1.0;
        label_buff_test.push(zero_vec);
    }
    
    let accurecy = network.test(&batch_buff_test, &label_buff_test);
    
    generation_counter += 1;
    
    println!("Generation {}, Error rate: {}", (generation_counter), accurecy);
};

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
    output_layer: Vec<Neuron>,
    learning_rate: f64
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
            output_layer: (output_layer_neuron_list),
            learning_rate: 1.0
        }
    }
    
    // runs a cykle, adds calculated factors to training_sum.
    fn train(&mut self, input_chunk: &Vec<Vec<f64>>, correct_chunk: &Vec<Vec<f64>>) {

        if input_chunk.is_empty() {
            return;
        }

        let mut average_gradient_h_w = Array2::<f64>::zeros((0,0));
        let mut average_gradient_h_b = Array2::<f64>::zeros((0,0));
        let mut average_gradient_o_w = Array2::<f64>::zeros((0,0));
        let mut average_gradient_o_b = Array2::<f64>::zeros((0,0));

        let mut iter = 0;


        for input in input_chunk {
            // hidden layer one a, output layer a
            let a_hidden_layer_one = self.feedforward(&input, hidden_layer_one);
            let a_output_layer = self.feedforward(&a_hidden_layer_one, output_layer);

            let input_matrix = &Array2::from_shape_vec((784,1), input.to_vec()).unwrap();
            // Generates the weight and bias matrix from the output layer
            let (weight_output_matrix, bias_output_matrix) = self.get_weight_bias_matrix(output_layer);
            let (weight_hidden_matrix, bias_hidden_matrix) = self.get_weight_bias_matrix(hidden_layer_one);
            
            // Backwards propagation, step one
            let gradient_output_correct = Array2::from_shape_vec((10,1),a_output_layer.iter().zip(correct_chunk.index(iter)).map(|(&x,&y)| (x - y) * x * (1.0 - x)).collect()).unwrap();

            let hidden_row = Array2::from_shape_vec((1, a_hidden_layer_one.len()), a_hidden_layer_one.clone(),).unwrap();

            let gradient_output_weights = gradient_output_correct.dot(&hidden_row);
            // Backwards propagation, step two

            let hidden_derivative = Array2::from_shape_vec((a_hidden_layer_one.len(), 1), a_hidden_layer_one.iter().map(|&x|x * (1.0 - x)).collect()).unwrap();

            let gradient_hidden_correct = weight_output_matrix.t().dot(&gradient_output_correct) 
                        * hidden_derivative;
            
            let gradient_hidden_weights = &gradient_hidden_correct.dot(&input_matrix.t());


            if iter == 0 {
            average_gradient_h_w = gradient_hidden_weights.to_owned();
            average_gradient_h_b = gradient_hidden_correct.to_owned();
            average_gradient_o_w = gradient_output_weights.to_owned();
            average_gradient_o_b = gradient_output_correct.to_owned();
            } else {
            average_gradient_h_w += gradient_hidden_weights;
            average_gradient_h_b += &gradient_hidden_correct;
            average_gradient_o_w += &gradient_output_weights;
            average_gradient_o_b += &gradient_output_correct;
            }
            iter += 1;
        }



        let scale = self.learning_rate / iter as f64;
        
        // Applying changes
        for neuron in &mut self.hidden_layer_one {
            let weight_row: Vec<f64> = average_gradient_h_w.row(0).to_vec();
            let bias_row: f64 = average_gradient_h_b.row(0).to_vec()[0];
            neuron.weights = neuron.weights.iter().zip(weight_row).map(|(x,y)| *x - scale * y).collect();
            neuron.bias -= scale * bias_row;

            average_gradient_h_b = average_gradient_h_b.slice(s![1.., ..]).to_owned();
            average_gradient_h_w = average_gradient_h_w.slice(s![1.., ..]).to_owned();
        }
        
        for neuron in &mut self.output_layer {
            let weight_row: Vec<f64> = average_gradient_o_w.row(0).to_vec();
            let bias_row: f64 = average_gradient_o_b.row(0).to_vec()[0];
            neuron.weights = neuron.weights.iter().zip(weight_row).map(|(x,y)| *x - scale * y).collect();
            neuron.bias -= scale * bias_row;

            average_gradient_o_b = average_gradient_o_b.slice(s![1.., ..]).to_owned();
            average_gradient_o_w = average_gradient_o_w.slice(s![1.., ..]).to_owned();
        }
        


    }

     
    // runs a cykle, adds calculated factors to training_sum.
    fn test(&mut self, input_chunk: &Vec<Vec<f64>>, correct_chunk: &Vec<Vec<f64>>) -> f64 {

        
        if input_chunk.is_empty() {
            panic!("error code -1.0");
        }

        let mut errors = 0;


        for (input, correct) in input_chunk.iter().zip(correct_chunk.iter()) {
            // hidden layer one a, output layer a
            let a_hidden_layer_one = self.feedforward(&input, hidden_layer_one);
            let a_output_layer = self.feedforward(&a_hidden_layer_one, output_layer);

            let predicted = argmax(&a_output_layer).expect("Empty network activation");
            let expected = argmax(correct).expect("Correct list was empty");
           

            if predicted != expected {
                errors += 1;
            }  
        };

        errors as f64 / input_chunk.len() as f64
    }


    fn get_weight_bias_matrix(&mut self, layer: layer) -> (Array2<f64>, Array2<f64>) {
        let (layer,(number,connections)) = match layer {
            hidden_layer_one => (&self.hidden_layer_one, (20,784)),
            hidden_layer_two => todo!(),
            output_layer => (&self.output_layer, (10,20))
        };

        let mut super_vector: Vec<f64> = Vec::new();
        let mut bias_vector: Vec<f64> = Vec::new();
        
        for neuron in layer {
            super_vector.extend_from_slice(&neuron.weights);
            bias_vector.push(neuron.bias);
        };
        
        let weight_matrix = Array2::from_shape_vec((number,connections), super_vector).unwrap();
        let bias_matrix = Array2::from_shape_vec((number,1), bias_vector).unwrap();

        (weight_matrix, bias_matrix)
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
            weight_list.push(rand::random_range(-0.1..=0.1));
        };
        let bias: f64 = 0.0;

        let mut new_neuron = Neuron { weights: (weight_list), bias: (bias) };
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
fn error_matrix(output: &Vec<f64>, correct: &Vec<f64>) -> Array2<f64>  {
    let cost = output.iter().zip(correct).map(|(x,y)|(x-y).powi(2));
    Array2::from_shape_vec((10, 1), cost.collect()).unwrap()
}

fn argmax(value_list: &Vec<f64>) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &value) in value_list.iter().enumerate() {
        if best.map_or(true, |x| value > value_list[x]) {
            best = Some(i);
        }
    }
    best
}