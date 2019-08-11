close all
clear variables
clc

%% XOR
bias = -1; % Fixed value setted permanently to -1
dataset = [0,0,bias; 0,1,bias; 1,0,bias; 1,1,bias]; % Input dataset (u)
row = length(dataset);
column = length(dataset(1,:));
output = [0, 1, 1, 0]; % Output valu(v: it's known cause it's a supervised learning problem)

%% Parameters and variables
v = 0; % Output of the binary classification
learning_rate = 0.3; % Higher it is, more swinging will be the convergence
epochs = 3000;
threshold_error = 0.001;
activationFunction = ActivationFunction.TLU;
error = zeros(1,epochs);
error = error + inf;

% Design of the net
number_of_input = 2 + 1; % Number of neurons in the input layer (+1 for bias) (u_i)
number_of_hidden_layer = 1; % Number of hidden layers
number_of_hidden_neuron = 2; % Number of neurons for each hidden layer x_j
number_of_output = 1; % Number of neurons in the output layer (v_k)

% Random initialization of the weights with value [-1,+1]
w_in_hid = -1 + (1+1)*rand(number_of_hidden_neuron, number_of_input); % Weigths between input and hidden layer
w_hid_out =  -1 + (1+1)*rand(number_of_output,number_of_hidden_neuron);

z_hidden = zeros(number_of_hidden_neuron, number_of_hidden_layer);
output_hidden =  zeros(number_of_hidden_layer, 1);
%% Algorithm
while error > threshold_error
    for epoch = 1:epochs
        disp('############ New epoch ###############')
        % Reordering and merging
        u = dataset;
        t = output;
        matrix_input_output = [u,t'];
        
        % Data shuffle
        matrix_input_output = Shuffle(matrix_input_output, row, column);
        
        % Obtaing shuffled data
        t = matrix_input_output(:,end)';
        u = matrix_input_output(:, 1:end-1);
              
        % For each input
        for i = 1:number_of_input
            output_hidden_layer = zeros(number_of_hidden_neuron, 1);
            input_vector = u(i,:);
            correct_output = t(i);
            
            % Feed forward
            
            % It would be necssary also a iteration over the number of
            % hidden layer
            % for i = 1:number_of_hidden_layer
            
            z_hidden = w_in_hid * input_vector';
            for j = 1:number_of_hidden_neuron
                switch activationFunction
                    case ActivationFunction.TLU
                        output_hidden_layer(j,1) = heaviside(z_hidden(j,1));
                    case ActivationFunction.Linear
                        output_hidden_layer(j,1) = z_hidden(j,1);
                    case ActivationFunction.Sigmoid
                        output_hidden_layer(j,1) = 1/ (1 + exp(-z_hidden(j,1)));
                end
            end
            
            z_output = w_hid_out * output_hidden_layer;
            
            switch activationFunction
                case ActivationFunction.TLU
                    % If more neurons in the output layer, create a vector
                    % for output_out_layer
                    output_out_layer = heaviside(z_output);
                case ActivationFunction.Linear
                    % If more neurons in the output layer, create a vector
                    % for output_out_layer
                    output_out_layer = z_output;
                case ActivationFunction.Sigmoid
                    % If more neurons in the output layer, create a vector
                    % for output_out_layer
                    output_out_layer = 1/ (1 + exp(-z_output));
            end
            
            % Back propagation
            % Made more generic (for more output neurons)
            error = abs(correct_output - output_out_layer);
            gradient_output = output_out_layer * (1 - output_out_layer) * error;
            gradient_hidden = zeros(number_of_hidden_neuron, 1);
            for j = 1:number_of_hidden_neuron
                v = output_hidden_layer(j,1);
                %iterate over each output nodes
                gradient_output_coeff = gradient_output * w_hid_out(1,j);
                gradient_hidden(j,1) = v * (1 - v) * gradient_output_coeff;
            end
            
            % Update weigths
            for k = 1:number_of_output
                w_hid_out(1,k) = w_hid_out(1,k) + learning_rate * gradient_output * output_hidden_layer(k,1);
            end
            for k = 1:number_of_hidden_neuron
                w_in_hid(:,k) = w_in_hid(:,k) +  learning_rate * gradient_hidden(k,1) * input_vector;
            end
        end
        
        
        
    end
end