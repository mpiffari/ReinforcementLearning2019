close all
clear variables
clc

%% Variables and parameters
numberOfState = 2; % Position and velocity
stateSpace = [0;0]; % Column vector

maximumAcceleration = 1;
stepDiscreteAcceleration = 0.01;
actionSpace = -maximumAcceleration : stepDiscreteAcceleration : maximumAcceleration;

% System physic parameters
h = 1; % Max height of the mountain [m]
L = 4; % Length of the valley
m = 1; % [kg]
g = 9.80665; % Gravity acceleration [m/s^2]

%Time
t_i = 0;
dt = 0.05;
t_f = 50;

now = 1;
after = 2;
position = [0, L/2];
velocity = [0 , 0.9];

epsilon = 0.3;

fig0 = figure;
%% Algorithm
for t = t_i : dt : t_f
    % Chose the action by greedy way
    time = clock;
    seed = time(6);
    rng(seed);
    action = rand();
    if action < epsilon
        a_t = -1 + (1+1)*rand();
    else
        a_t = 1;
    end
    
    %% Matrix calculation
    %     first_row = (1 / A) * ( B * velocity(now) + C);
    %     second_row = a_t / D;
    %     stateSpace = stateSpace + t * [first_row; second_row]
    
    %% Calculation with physic equation
    position(now) = position(after);
    velocity(now) = velocity(after);
    x = position(now);
    
    params = Parameters(x, h, m, L);
    A = params(1);
    B = params(2);
    C = params(3);
    D = params(4);
    
    velocity(after) = velocity(now) + dt * (1/A)*(m * (a_t/D) - C - B * (velocity(now))^2);
    position(after) = x + dt * velocity(now);
    if position(after) < 0
        position(after) = 0;
        velocity(after) = 0;
    elseif position(after) >= L
        disp('####### YEEE ########');
        break
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
