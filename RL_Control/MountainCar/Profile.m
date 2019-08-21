function y = Profile(x, L, h)
    % Profile of the mountain
    y = 0.5 * h * (1 + cos(2 * pi * x / L));
end

