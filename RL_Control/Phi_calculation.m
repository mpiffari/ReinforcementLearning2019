function phi = Phi_calculation(state, mu, sigma)
    x = state(1,1);
    v = state(2,1);
    
    %figLocal = figure();
    %figure(figLocal);
    %hold on
    %scatter(mu(:,1),mu(:,2), 'filled');
    
    phi = zeros(length(mu),1);
    % For each centrum
    for i = 1:length(mu)
       c_x = mu(i,1);
       c_v = mu(i,2);
       sigma_x = sigma(i,1);
       sigma_v = sigma(i,2);
       
       coeff_1 = (x - c_x)^2 / (2 * sigma_x^2);
       coeff_2 = (v - c_v)^2 / (2 * sigma_v^2);
       phi(i,:) = exp(-(coeff_1 + coeff_2));
    end
end

