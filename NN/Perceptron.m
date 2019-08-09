close all
clear variables
clc

%% Import data from excel file
% Import Data in the Home menu
irisDataSet = importdata('Iris_Dataset_reduced.xlsx');
dataset = irisDataSet.data;
label = irisDataSet.textdata(2:end,3);
bias = -1; % Fixed value setted permanently to -1
row = length(dataset);
column = length(dataset(1,:));

%% Normal code
%bias = -1; % Fixed value setted permanently to -1
%dataset = [0,0,bias; 0,1,bias; 1,0,bias; 1,1,bias]; % Input dataset (u)
%row = length(dataset);
%column = length(dataset(1,:));
%% AND
%output = [0, 0, 0, 1]; % Output valu(v: it's known cause it's a supervised learning problem)

%% OR
%output = [0, 1, 1, 1]; % Output value

%% Parameters and variables
v = 0; % Output of the binary classification
learning_rate = 0.1; % Higher it is, more swinging will be the convergence
w = -1 + (1+1)*rand(1,column); % Random initialization of the weights with value [-1,+1]
epochs = 100;
threshold_error = 0.001;
activationFunction = ActivationFunction.TLU;
error = zeros(1,epochs);
error = error + inf;
fig1 = figure;
fig2 = figure;

%% Algorithm
while error > threshold_error
    for epoch = 1:epochs
        % Reordering and merging
        u = dataset;
        t = output;
        matrix_input_output = [u,t'];
        
        % Data shuffle
        matrix_input_output = Shuffle(matrix_input_output);
        
        % Obtaing shuffled data
        t = matrix_input_output(:,end)';
        u = matrix_input_output(:, 1:end-1);
        
        % For each possible  of the dataset
        for i = 1:row
            input_vector = u(i,:);
            correct_output = t(i);
            z = input_vector * w';
            
            switch activationFunction
                case ActivationFunction.TLU
                    output_layer = heaviside(z);
                    error(1,epoch) = correct_output - output_layer;
                    w = w + (0.5 * learning_rate) * (error(1,epoch) * input_vector')';
                case ActivationFunction.Linear
                    output_layer = z;
                    error(1,epoch) = correct_output - output_layer;
                    w = w + learning_rate * (error(1,epoch) * input_vector')';
                case ActivationFunction.Sigmoid
                    output_layer = 1/ (1 + exp(-z));
                    error(1,epoch) = output_layer * (1 - output_layer);
                    w = w + learning_rate * (error(1,epoch) * input_vector')';
            end
                     
            % Dynamic plot
            w1= w(1);
            w2= w(2);
            wbias= w(3);
            
            x = linspace(-0.5,1.5, 1000); % Adapt n for resolution of graph
            y = -(w1/w2)*x -(bias*wbias)/w2;
 
            figure(fig1);
            clf(fig1)
            hold on
            plot(x,y);
            ylim([-5 5])
            hold on
            scatter(input_vector(1), input_vector(2), 'filled')
        end
    end
end

hold on
scatter(dataset(:,1), dataset(:,2));
xlabel('First gate');
ylabel('Second gate');
% Error
figure(fig2)
x = -0.5:1:epochs-1;
scatter(x,error);

disp('End')