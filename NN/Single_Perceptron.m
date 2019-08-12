close all
clear variables
clc
load fisheriris

%% Iris dataset
irisDataSet = meas(:,1:2); % Only two features
% Keep only two species linearly separable
irisDataSet(100:end,:) = [];
species(100:end,:) = [];
y = categorical(species);

label = zeros(length(y),1);
for i = 1:length(y)
    if y(i) == 'setosa'
        label(i,1) = 0;
    else
        label(i,1) = 1;
    end
end
row = length(irisDataSet);
column = length(irisDataSet(1,:)) + 1; % +1 for the bias coefficient
bias = -1; % Fixed value setted permanently to -1
mat = repmat(bias, row);
columnBias = mat(:,1);
dataset = [irisDataSet(:,1:2), columnBias]; % Bias coeff always in the last position
output = label;

gscatter(irisDataSet(:,1),irisDataSet(:,2),species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
%% AND
%bias = -1; % Fixed value setted permanently to -1
%dataset = [0,0,bias; 0,1,bias; 1,0,bias; 1,1,bias]; % Input dataset (u)
%row = length(dataset);
%column = length(dataset(1,:));
%output = [0, 0, 0, 1]; % Output valu(v: it's known cause it's a supervised learning problem)

%% OR
%bias = -1; % Fixed value setted permanently to -1
%dataset = [0,0,bias; 0,1,bias; 1,0,bias; 1,1,bias]; % Input dataset (u)
%row = length(dataset);
%column = length(dataset(1,:));
%output = [0, 1, 1, 1]; % Output value

%% Parameters and variables
v = 0; % Output of the binary classification
learning_rate = 0.3; % Higher it is, more swinging will be the convergence
w = -1 + (1+1)*rand(1,column); % Random initialization of the weights with value [-1,+1]
epochs = 50;
threshold_error = 0.01;
activationFunction = ActivationFunction.TLU;
error = zeros(1,epochs);
error = error + inf;
fig1 = figure;
%% Algorithm
while error > threshold_error
    for epoch = 1:epochs
        disp('############ New epoch ###############')
        % Reordering and merging
        u = dataset;
        t = output;
        matrix_input_output = [u,t];
        
        % Data shuffle
        matrix_input_output = Shuffle(matrix_input_output, row, column);
        
        % Obtaing shuffled data
        t = matrix_input_output(:,end)';
        u = matrix_input_output(:, 1:end-1);
        
        % For each possible data of the dataset
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
                    error(1,epoch) = 0.5 * (correct_output - output_layer)^2;
                    w = w + learning_rate * (output_layer * (1 - output_layer) * (correct_output - output_layer) * input_vector')';
            end
               
            disp(error(1,epoch));
            % Dynamic plot
            w1= w(1);
            w2= w(2);
            wbias= w(3);
            
            x = linspace(2,10, 1000); % Adapt n for resolution of graph
            y = -(w1/w2)*x -(bias*wbias)/w2;
 
            figure(fig1);
            clf(fig1)
            hold on
            plot(x,y);
            ylim([-10 10])
            hold on
            scatter(input_vector(1), input_vector(2), 'filled')
        end
    end
end

clf(fig1)
% Bound equation
w1= w(1);
w2= w(2);
wbias= w(3);
x = linspace(2,10, 1000); % Adapt n for resolution of graph
y = -(w1/w2)*x -(bias*wbias)/w2;
figure(fig1);
plot(x,y);
hold on
gscatter(irisDataSet(:,1),irisDataSet(:,2),species,'rgb','osd');
xlabel('Sepal Length Cm');
ylabel('SepalWidthCm');

% Error
fig2 = figure;
figure(fig2)
x = -0.5:1:epochs-1;
scatter(x,error);

disp('End')