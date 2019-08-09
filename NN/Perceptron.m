

bias = 0;
u = [bias, 0,0; bias,0,1; bias,1,0; bias,1,1];
v = 0;
t = [0, 0, 0, 1];
weigths = -1 + (1+1)*rand(1,2);
w = [-1, weigths];
error = inf;
epochs = 1000;

while error > 0.001
    u
    for epoch = 1:epochs
        matrix_input_output = [u,t'];
        Shuffle(matrix_input_output);
        matrix_input_output
        t = matrix_input_output(:,end)';
        u = matrix_input_output(:, 1:end-1);
        for i = 1:4
            input_vector = u(i,:);
            correct_output = t(i);
        end
    end
end
