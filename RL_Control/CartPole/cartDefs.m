%% Declare the parameters of the cart and simulation as global
global cart

%% Parameters of the cart
cart.M = 5;     % mass

%% Parameters of the pole
cart.m = 0.5;    % mass
cart.l = 0.75;   % length

%% Other parameters
cart.g = 9.81;   % Gravity

cart.dT = 0.05; 
cart.tf = 10; % Timeout
cart.numberOfState = 4;

cart.maximumAngle = pi/6;
cart.maximumAngularVelocity = sqrt((1-cos(cart.maximumAngle)) * cart.g / cart.l);
cart.initialAngle = 0; % TODO: randomize
cart.initialAngularVelocity = 0;

%% Parameters of the algorithm
% RBF parameters
cart.discret_angle = 6;
cart.discret_angularVelocity = 6;
cart.number_of_centrum = cart.discret_angle * cart.discret_angularVelocity;
cart.mu = zeros(cart.number_of_centrum, 2);
cart.sigma = zeros(cart.number_of_centrum, 2);

cart.fig1 = figure;


