function Q = FunctionApproximator(w,state, mu, sigma)
    % W = vector of weigth, positive and negative = [positive, neagtive]
    phi = Phi_calculation(state, mu, sigma); % phi = phi_positive = phi_negative
    Q = w * phi;
end

