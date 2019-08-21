function value = Bernoulli(p, dimension)
    pd = makedist('Binomial','N',1,'p',p);
    value = - 1 + 2 * random(pd,1,dimension);
end

