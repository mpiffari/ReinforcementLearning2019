close all
clear variables
clc

%% Notes
% No need to use k-means to determine the center mu of each RBF

%% Variables and parameters
% System physic parameters
h = 1; % Max height of the mountain [m]
L = 4; % Length of the valley
m = 1; % [kg]
g = 9.81; % Gravity acceleration [m/s^2]
maximumVelocity = sqrt(2 * g * h); % [m/s]
maximumAcceleration = 4; % [m/s]
% stepDiscreteAcceleration = 0.01;

% Plot parameters
max = h;
min = -1;
x_mountains = min: 0.01 : L - min;
y_mountains = zeros(length(x_mountains),1);
for i = 1:length(x_mountains)
    y_mountains(i,1) = Profile(x_mountains(i),L,h);
end

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
epsilon_0 = 0.8;
gamma = 1;
learning_rate = 0.1; % Alpha
isTerminalState = 0;

%Time
number_of_episode = 3000;
t_i = 0; % [s]
dt = 0.05; % [s]
timeout = +inf; % [s] = final execution time

% Random seed
rng(50);
% time = clock;
% seed = time(6);
% rng(seed);


% Misc parameters
now = 1;
after = 2;
position = [L/2, 0];
velocity = [0, 0];
reward = 0;
plot_rewards = zeros(number_of_episode, 1);
index = 1;
plotActive = 0;
debugActive = 0;
fig0 = figure;
fig1 = figure;
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
scatter(mu(:,1),mu(:,2));
xlabel('Car position')
ylabel('Car velocity')
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
    fprintf('Episode number: %f\n',episode);
    position = [L/2, 0];
    velocity = [0, 0];
    stateSpace(:,1) = [position(now); velocity(now)]; % Initialization of the state
    sign_of_acc = sign(-1 + (1+1)*rand()); % Random initialization of the very first action
    a_t = sign_of_acc * maximumAcceleration;
    t = t_i;
    isTerminal = 0;
    index = 1;
    rewards = [];
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
        reward = RewardCarMountain(position(after), L);
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
        Q_positive = FunctionApproximator(w_hid_out_positive, [position(after); velocity(after)], mu, sigma);
        Q_negative = FunctionApproximator(w_hid_out_negative, [position(after); velocity(after)], mu, sigma);
        if Q_negative > Q_positive
            nextAction = - maximumAcceleration;
        else
            nextAction = maximumAcceleration;
        end
        
        epsilon = epsilon_0 / episode;
        if rand() < epsilon
            sign_of_acc = sign(-1 + (1+1)*rand());
            nextAction = sign_of_acc * maximumAcceleration;
        end
        
        %%%%%%%%%%%%%% Weights update %%%%%%%%%%%%%%%%%%%%
        gradient_Q = Phi_calculation([position(now); velocity(now)], mu, sigma);
        if nextAction > 0
            Q_t1 = FunctionApproximator(w_hid_out_positive, [position(after); velocity(after)], mu, sigma);
            Q_t = FunctionApproximator(w_hid_out_positive, [position(now); velocity(now)], mu, sigma);
            % Keep it as a vector (output neuron x hidden neuron)
            w_hid_out_positive = w_hid_out_positive + learning_rate * (reward + gamma * Q_t1 - Q_t) * gradient_Q';
        else
            Q_t1 = FunctionApproximator(w_hid_out_negative, [position(after); velocity(after)], mu, sigma);
            Q_t = FunctionApproximator(w_hid_out_negative, [position(now); velocity(now)], mu, sigma);
            % Keep it as a vector (output neuron x hidden neuron)
            w_hid_out_negative = w_hid_out_negative + learning_rate * (reward + gamma * Q_t1 - Q_t) * gradient_Q';
        end
        
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

