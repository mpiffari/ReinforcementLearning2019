function [A,B,C,D] = Parameters(x,h,m,L)
    coeff_1 = (h^2 / L^2) * pi^2;
    coeff_2 = (h^2 / L^3) * pi^3;
    arg = 2 * pi * x / L;
    g = 9.80665; % [m/s^2]
    
    A = m * (1+ coeff_1 * (sin(arg))^2);
    B = m * coeff_2 * sin (2 * arg);
    C = - ((m * g * h)/L) * pi * sin(arg);
    D = sqrt(1 + coeff_1 * (sin(arg/2))^2);
end

