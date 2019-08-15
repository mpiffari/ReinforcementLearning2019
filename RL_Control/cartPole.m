close all
clear variables
clc

%% Notes
% No need to use k-means to determine the center mu of each RBF

%% Variables and parameters
global cart;
% System physic parameters
g = cart.g; % Gravity acceleration [m/s^2]
maximumAngle = cart.zmax; % [m/s]
maximumAngularVelocity = cart.wmax; % [m/s]
% stepDiscreteAcceleration = 0.01;

% State space
numberOfState = 2; % Position and velocity
stateSpace = [0;0]; % Column vector
% Action space --> we can apply any action
actionSpace = [- inf, + inf];

% RBF parameters
discret_angle = 6;
discret_angularVelocity = 6;

% SPSA parameters
alpha = 
gamma = 

%Time
number_of_episode = 3000;
t_i = 0; % [s]
dt = cart.dt; % [s]
timeout = cart.tf; % [s] = final execution time

% Random seed
% time = clock;
% seed = time(6);
rng(55);

% Misc parameters
now = 1;
after = 2;
angle = [0, 0];
angularVelocity = [0, 0];
reward = 0;
plot_rewards = zeros(number_of_episode, 1);
index = 1;
plotActive = 0;
debugActive = 0;
fig0 = figure;
fig1 = figure;
%% Center and variance computation
row = 1;
angle_step = L / (discret_angle - 1);
angularVelocity_step = (maximumAngularVelocity * 2) / (discret_angularVelocity - 1);
number_of_centrum = discret_position * discret_velocity;
sigma_position = position_step / sqrt(2 * number_of_centrum);
sigma_velocity = velocity_step / sqrt(2 * number_of_centrum);

mu = zeros(number_of_centrum, 2);
sigma = zeros(number_of_centrum, 2); % [sigma_position, sigma_velocity]
for i= 0 : position_step : L
    for j = - maximumVelocity : velocity_step : maximumVelocity
        mu(row, :) = [i,j];
        row = row + 1;
    end
end

for i = 1 : number_of_centrum
    sigma(i,:) = [sigma_position, sigma_velocity];
end
%% Design of the net
number_of_input = 2; % Number of neurons in the input layer (u_i)
number_of_hidden_layer = 1; % Number of hidden layers
number_of_hidden_neuron = number_of_centrum; % Number of neurons for each hidden layer x_j
number_of_output = 1; % Number of neurons in the output layer (v_k)

% Random initialization of the weights with value [-1,+1]
% Row: hidden neuron - bias neuron, Column: input neuron
w_in_hid = ones(number_of_hidden_neuron, number_of_input); % Weigths between input and hidden layer: ALWAYS FIXED TO 1
% Row: output neuron, Column: hidden neuron
w_hid_out_positive =  -1 + (1+1)*rand(number_of_output, number_of_hidden_neuron);
w_hid_out_negative =  -1 + (1+1)*rand(number_of_output, number_of_hidden_neuron);

phi_positive = rand(number_of_output, number_of_hidden_neuron);
phi_negative = rand(number_of_output, number_of_hidden_neuron);

%% Algorithm

for episode = 1:number_of_episode
       
    for NH
    
    while isTerminalState == 0 && t < timeout
        %%%%%%%%%%%%%% Print debug %%%%%%%%%%%%%%%%%%%%
        if debugActive == 1
            fprintf('==============================================\n')
            fprintf('Current state: pos = %f, vel = %f \n', position(now),velocity(now));
            fprintf('Current action: %f \n', a_t);
            fprintf('==============================================\n')
        end
        %%%%%%%%%%%%%% Take action %%%%%%%%%%%%%%%%%%%%
        x = position(now);
        params = Parameters(x, h, m, L);
        A = params(1);
        B = params(2);
        C = params(3);
        D = params(4);
        
        velocity(after) = velocity(now) + dt * (1/A)*(m * (a_t/D) - C - B * (velocity(now))^2);
        position(after) = x + dt * velocity(now);
        reward = Reward(position(now), L);
        rewards(index,:) = reward;
        index = index + +1;
        
        %%%%%%%%%%%%%% Check new state %%%%%%%%%%%%%%%%%%%%
        if position(after) < 0
            disp('####### LEFT REACHED ##########')
            position(after) = 0;
            velocity(after) = 0;
            break
        elseif position(after) >= L
            disp('TARGET')
            % Q value calculation: Q = output of the RBF net = linear
            % combination of weigth and phi function
            % Phi function calculation: phi_positive = phi_negative
            phi = Phi_calculation(stateSpace, mu, sigma);
            gradient_Q = phi;
            
            %Update weigth
            if a_t > 0
                % Update positive weigths
                Q_t = FunctionApproximator(w_hid_out_positive, [position(now); velocity(now)], mu, sigma);
                % Keep it as a vector (output neuron x hidden neuron)
                w_hid_out_positive = w_hid_out_positive + learning_rate * (reward - Q_positive) * gradient_Q';
            else
                % Update negative weigths
                Q_t = FunctionApproximator(w_hid_out_negative, [position(now); velocity(now)], mu, sigma);
                % Keep it as a vector (output neuron x hidden neuron)
                w_hid_out_negative = w_hid_out_negative + learning_rate * (reward - Q_negative) * gradient_Q';
            end
            isTerminal = 1;
            break
        end
        
        %%%%%%%%%%%%%% Choose next action %%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%% Weights update %%%%%%%%%%%%%%%%%%%%

        
        %%%%%%%%%%%%%% State, action and time update %%%%%%%%%%%%%%%%%%%%
        a_t = nextAction;
        stateSpace(1,1) = position(after);
        stateSpace(2,1) = velocity(after);
        position(now) = position(after);
        velocity(now) = velocity(after);
        t = t + dt;
        if t >= timeout
            disp('####### TIMEOUT ##########')
        end
        %%%%%%%%%%%%%% Plot %%%%%%%%%%%%%%%%%%%%
        if plotActive == 1
            figure(fig0);
            clf(fig0)
            hold on
            plot(x_mountains, y_mountains);
            y_point = Profile(position(after),L,h);
            y_target = max;
            scatter(position(after), y_point, 'filled');
            scatter(L, y_target, '*');
            ylim([0 max+1]);
            xlim([min L+1]);
        end
    end
    plot_rewards(episode, :) = mean(rewards);
end
figure();
x = 1: 1 : number_of_episode;
plot(x, plot_rewards);

