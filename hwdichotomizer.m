function hwdichotomizer=hwdichotomizer(X)
out=[];

[h,w]=size(X);

for i=1:h
    % returns bigges value in row and its index(class)
    [val,idx]=max(X(i,1:2));

    out=[out;idx];
end
hwdichotomizer=out;
end