% Declare the parameters of the cart and simulation as global
global cart

% Parameters of the cart
cart.M = 5;     % mass

% Parameters of the pole
cart.m = 0.5;    % mass
cart.l = 0.75;   % length

% Other parameters
cart.g = 9.8;   % Gravity

cart.dT = 0.05; 
cart.tf = 10;

cart.zMax = pi/6;
cart.wMax = sqrt((1-cos(cart.zMax)) * cart.g / cart.l);
