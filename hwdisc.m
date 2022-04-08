%%Discriminant function
function hwdisc=hwdisc(p,X)
out=[];
[h,w]=size(p);
[n,d] =size(X);
    
for i=1:h        
    %% Discriminant for class 3     
    %calculate covariance from class 1 data
    sigma_c1=cov(X(1:10));
    %calculate mean from class 1 data
    mu_c1=cov(X(1:10));
    
    dc1 =(-1/2)*(p(i)-mu_c1)*(sigma_c1^(-1))*(p(i)-mu_c1)'+(-d/2)*log(2*pi)+(-1/2)*log(det(sigma_c1))+log(1/2);
    
    %% Discriminant for class 2
    %calculate covariance from class 2 data
    sigma_c2=cov(X(11:20));
    %calculate mean from class 2 data
    mu_c2=cov(X(11:20));
        
    dc2 =(-1/2)*(p(i)-mu_c2)*(sigma_c2^(-1))*(p(i)-mu_c2)'+(-d/2)*log(2*pi)+(-1/2)*log(det(sigma_c2))+log(1/2);

    %% Discriminant for class 3 
    %calculate covariance from class 3 data
    sigma_c3=cov(X(21:30));
    %calculate mean from class 3 data
    mu_c3=cov(X(21:30));
    
    %removed prior because ln(0)=infinity
    dc3 = (-1/2)*(p(i)-mu_c3)*(sigma_c3^(-1))*(p(i)-mu_c3)'+(-d/2)*log(2*pi)+(-1/2)*log(det(sigma_c3));

    out=[out;dc1 ,dc2 ,dc3];
end

% return a 1x3 vector output
hwdisc=out;
end