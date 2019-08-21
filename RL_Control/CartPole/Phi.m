function value = Phi(x,v,c_x,c_v,sigma_x, sigma_v)
    coeff_1 = (x - c_x)^2/ (2 * sigma_x);
    coeff_2 = (v - c_v)^2 / (2 * sigma_v);
    value = exp(-(coeff_1 + coeff_2));
end

