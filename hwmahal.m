function hwmahal=hwmahal(x,mu,sig)
% calculates mahalanobis distance for given point x, mean and covariance
hwmahal=sqrt((x-mu)'*sig^(-1)*(x-mu));
end