function dX = cartDynamics(X, u)

% State: X =[x th v w]

global cart
m = cart.m;
M = cart.M;
l = cart.l;
g = cart.g;

q = X(1:2);
dq = X(3:4);

H = [m + M, -m * l * cos(q(2)); ...
    -m * cos(q(2)), m * l];

% C(q, dq):
C = [0, m * l * sin(q(2)) * dq(2); ...
    0, 0];
% G(q):
G = [0;...
    -m * g * sin(q(2))];
% and B:
B = [1;...
    0];
ddq = inv(H) * (B * u - C * dq - G);
dX = [dq; ddq];
end