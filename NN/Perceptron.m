T = now;
rng(T)

bias = 0;
u = [bias, 0,0; bias,0,1; bias,1,0; bias,1,1]
v = 0;
t = [0, 0, 0, 1];
weigths = -1 + (1+1)*rand(1,2)
w = [-1, weigths]