function hwgennormaldist=hwgennormaldist(mu,sig,count)
% Generates random samples according to normal distribution
hwgennormaldist=mvnrnd(mu,sig,count);
end