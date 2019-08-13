close all
clear variables
clc

time = clock;
seed = time(6);
rng(seed);

%% Notes
% No need to use k-means to determine the center mu of each RBF

%% Variables and parameters
% System physic parameters
h = 1; % Max height of the mountain [m]
L = 100; % Length of the valley
m = 1; % [kg]
g = 9.80665; % Gravity acceleration [m/s^2]
maximumVelocity = sqrt(2 * g * h); % [m/s]
maximumAcceleration = 4; % [m/s]
% stepDiscreteAcceleration = 0.01;

% State space
numberOfState = 2; % Position and velocity
stateSpace = [0;0]; % Column vector
% actionSpace = -maximumAcceleration : stepDiscreteAcceleration :
% maximumAcceleration; --> continuos action space
actionSpace = [- maximumAcceleration, maximumAcceleration];

% RBF parameters
discret_position = 5;
discret_velocity = 5;

% SARSA parameters
epsilon_0 = 0.9;
gamma = 1;
learning_rate = 0.4; % Alpha
isTerminalState = 0;

%Time
number_of_episode = 100;
t_i = 0; % [s]
dt = 0.05; % [s]
timeout = 10; % [s] = final execution time

now = 1;
after = 2;
position = [L/2, 0];
velocity = [0, 0];
reward = [0, 0];

fig0 = figure;

%% Center and variance computation
row = 1;
position_step = L / (discret_position - 1);
velocity_step = (maximumVelocity * 2) / (discret_velocity - 1);
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
scatter(mu(:,1),mu(:,2), 'filled');

%% Design of the net
number_of_input = 2; % Number of neurons in the input layer (u_i)
number_of_hidden_layer = 1; % Number of hidden layers
number_of_hidden_neuron = number_of_centrum; % Number of neurons for each hidden layer x_j
number_of_output = 1; % Number of neurons in the output layer (v_k)

% Random initialization of the weights with value [-1,+1]
% Row: hidden neuron - bias neuron, Column: input neuron
w_in_hid = ones(number_of_hidden_neuron, number_of_input); % Weigths between input and hidden layer: ALWAYS FIXED TO 1
% Row: output neuron, Column: hidden neuron
w_hid_out_positive =  -1 + (1+1)*rand(number_of_output, number_of_hidden_neuron + 1);
w_hid_out_negative =  -1 + (1+1)*rand(number_of_output, number_of_hidden_neuron + 1);

phi_positive = rand(number_of_output, number_of_hidden_neuron + 1);
phi_negative = rand(number_of_output, number_of_hidden_neuron + 1);

%% Algorithm

for episode = 1:number_of_episode
    stateSpace(:,1) = [position(now); velocity(now)]; % Initialise state
    sign_of_acc = sign(-1 + (1+1)*rand()); % Random initialization of the very first action
    a_t = sign_of_acc * maximumAcceleration;
    t = t_i;
    isTerminal = 0;
    while isTerminalState == 0 || t < timeout
        %%%%%%%%%%%%%% Take action %%%%%%%%%%%%%%%%%%%%
        x = position(now);
        params = Parameters(x, h, m, L);
        A = params(1);
        B = params(2);
        C = params(3);
        D = params(4);
        
        velocity(after) = velocity(now) + dt * (1/A)*(m * (a_t/D) - C - B * (velocity(now))^2);
        position(after) = x + dt * velocity(now);
        reward(after) = Reward(position(now), L);
        
        if position(after) < 0
            position(after) = 0;
            velocity(after) = 0;
        elseif position(after) >= L
            disp('####### YEEEE ##########')
            % Q value calculation: Q = output of the RBF net = linear
            % combination of weigth and phi function
            % Phi function calculation: phi_positive = phi_negative
            phi_positive = Phi_calcultation(stateSpace, mu, sigma);
            phi_negative = phi_positive;
            
            %Update weigth
            if a_t > 0
                % Update positive weigths
                w_hid_out_positive = w_hid_out_positive + learning_rate
            else
                % Update negative weigths
                w_hid_out_negative = w_hid_out_negative + learning_rate
            end
            
            isTerminal = 1;
            break
        else
            
        
        t = t + dt;
    end
    
end

%% Old alg
for t = t_i : dt : t_f
    
    % Take the the action :  greedy way
    
    % phi calculation based on actual position
    
    
    Q_minus = 0 * w_hid_out_negative;
    Q_plus = 0 * w_hid_out_positive;
    
    if Q_minus > Q_plus
        a_t = - maximumAcceleration;
    else
        a_t = maximumAcceleration;
    end
    
    action = rand();
    epsilon = epsilon_0 / t;
    if action < epsilons
        sign_of_acc = sign(-1 + (1+1)*rand());
        a_t = sign_of_acc * maximumAcceleration;
    end
    
    %% Calculation with physic equation
    
    
    %reward calc
    % if s_(t+1) is terminal
    
    if position(after) < 0
        position(after) = 0;
        velocity(after) = 0;
    elseif position(after) >= L
        disp('####### YEEE ########');
        break
    else
        % Update weigth
    end
    stateSpace(1,1) = position(after);
    stateSpace(2,1) = velocity(after);
    
    %% Plot
    figure(fig0);
    clf(fig0)
    hold on
    
    max = h;
    min = -1;
    x_mountains = min: 0.01 : L;
    y_mountains = zeros(length(x_mountains),1);
    for i = 1:length(x_mountains)
        y_mountains(i,1) = Profile(x_mountains(i),L,h);
    end
    plot(x_mountains, y_mountains);
    y_point = Profile(position(after),L,h);
    y_target = max;
    scatter(position(after), y_point, 'filled');
    scatter(L, y_target, '*');
    ylim([0 max+1]);
    xlim([min L+1]);
end