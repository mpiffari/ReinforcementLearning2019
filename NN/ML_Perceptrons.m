close all
clear variables
clc

time = clock;
seed = time(6);
rng(seed);
%% XOR
bias = -1; % Fixed value setted permanently to -1
dataset = [0,0,bias; 0,1,bias; 1,0,bias; 1,1,bias]; % Input dataset (u)
row = length(dataset);
column = length(dataset(1,:));
output = [0, 1, 1, 0]; % Output value(v: it's known cause it's a supervised learning problem)
%% Parameters and variables
learning_rate = 0.3; % Higher it is, more swinging will be the convergence
epochs = 3500;
threshold_error = 0.00001;
activationFunction = ActivationFunction.Sigmoid;
errors = zeros(1,epochs);
errors = errors + inf;
error = inf;

% Design of the net
number_of_input = 2 + 1; % Number of neurons in the input layer (+1 for bias) (u_i)
number_of_hidden_layer = 1; % Number of hidden layers
number_of_hidden_neuron = 2+1; % Number of neurons for each hidden layer x_j
number_of_output = 1; % Number of neurons in the output layer (v_k)

% Random initialization of the weights with value [-1,+1]
% Row: hidden neuron - bias neuron, Column: input neuron
w_in_hid = -1 + (1+1)*rand(number_of_hidden_neuron - 1, number_of_input); % Weigths between input and hidden layer
% Row: output neuron, Column: hidden neuron
w_hid_out =  -1 + (1+1)*rand(number_of_output, number_of_hidden_neuron);

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
              
        % For each input of the dataset
        for i = 1:row
            output_hidden_layer = zeros(number_of_hidden_neuron, 1);
            input_vector = u(i,:);
            correct_output = t(i);
            
            % Feed forward
            
            % It would be necssary also a iteration over the number of
            % hidden layer
            % for i = 1:number_of_hidden_layer
            
            z_hidden = w_in_hid * input_vector';
            for j = 1:number_of_hidden_neuron - 1 % minus one for the bias
                switch activationFunction
                    case ActivationFunction.TLU
                        output_hidden_layer(j,1) = heaviside(z_hidden(j,1));
                    case ActivationFunction.Linear
                        output_hidden_layer(j,1) = z_hidden(j,1);
                    case ActivationFunction.Sigmoid
                        output_hidden_layer(j,1) = 1/ (1 + exp(-z_hidden(j,1)));
                end
            end
            
            output_hidden_layer(number_of_hidden_neuron,1) = bias;
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
            errors(1,epoch)= error;
            gradient_output = output_out_layer * (1 - output_out_layer) * error;
            gradient_hidden = zeros(number_of_hidden_neuron, 1);
            for j = 1:number_of_hidden_neuron
                v = output_hidden_layer(j,1);
                %iterate over each output nodes
                gradient_output_coeff = gradient_output * w_hid_out(1, j);
                gradient_hidden(j,1) = abs(v * (1 - v) * gradient_output_coeff);
            end
            
            % Update weigths
            for k = 1:number_of_output
                % For each row
                w_hid_out(k,:) = w_hid_out(k, :) + learning_rate * gradient_output * output_hidden_layer(k,1);
            end
            for k = 1:number_of_hidden_neuron - 1 % minus 1 for the bias
                % For each row
                w_in_hid(k,:) = w_in_hid(k,:) + learning_rate * gradient_hidden(k,1) * input_vector(1,k);
            end
        end       
    end
end

disp('############ END EPOCHS ###############');

% Bound equation
% Number of bound = number of nodes in the hidden layer
w1_1= w_in_hid(1,1);
w1_2= w_in_hid(1,2);
wbias_1= w_in_hid(1,3);

w2_1= w_in_hid(2,1);
w2_2= w_in_hid(2,2);
wbias_2= w_in_hid(2,3);

x = linspace(-0.5,1.5, 1000); % Adapt n for resolution of graph
y_1 = -(w1_1/w1_2)*x -(bias * wbias_1)/w1_2;
y_2 = -(w2_1/w2_2)*x -(bias * wbias_2)/w2_2;
figure();
plot(x,y_1);
hold on
plot(x,y_2);
ylim([-0.5 1.5]);
xlim([-0.5 1.5]);
scatter(dataset(:,1),dataset(:,2),'filled');
xlabel('Gate A');
ylabel('Gate B');