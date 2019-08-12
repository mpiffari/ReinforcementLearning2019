close all
clear variables
clc

%% Variables and parameters
numberOfState = 2; % Position and velocity
stateSpace = [0, 0]; 

maximumAcceleration = 10;
stepDiscreteAcceleration = 0.01;
actionSpace = -maximumAcceleration : stepDiscreteAcceleration : maximumAcceleration;

h = 10; % Max height of the mountain [m]
L = 100; % Length of the valley
noise = 0;

%Time
t_i = 0;
dt = 0.01;
t_f = 5;

now = 0;
after = 1;
position = [0 , 0];
velocity = [0 , 0];

%% Algorithm
for t = t_i : dt : t_f
    % Chose the action
    a_t = 1;
    
    params = Parameters(position(now), h, m, L);
    A = params(1);
    B = params(2);
    C = params(3);
    D = params(4);
    
    position(after) = position(now) + (t / A) * ( B * velocity(now) + C);
    velocity(after) = velocity(now) + t * a_t / D; 
end
    