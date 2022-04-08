%% Read data from class files
clear; clc; close all;

class1=readtable('class1.txt');
class2=readtable('class2.txt');
class3=readtable('class3.txt');

[h1, w1]=size(class1);
[h2, w2]=size(class2);
[h3, w3]=size(class3);

c1=table2array(class1);
c2=table2array(class2);
c3=table2array(class3);

y1(1:h1,1)=1;
y2(1:h1,1)=2;
y3(1:h1,1)=3;


X=[c1;c2;c3];
Y=[y1;y2;y3];

%% using only x1
disc=hwdisc(X(:,1),X);
pred=hwdichotomizer(disct);

% calculates accuracy using tp+tn/tp+tn+fp+fn
miss_classified = 1-sum(Y == pred,'all')/numel(pred);

%% using x1 and x2
disc=hwdisc(X(:,1:2),X);
pred=hwdichotomizer(disc);

% calculates loss
miss_classified = 1-sum(Y == pred,'all')/numel(pred);

%% using all features
disc=hwdisc(X(:,:),X);
pred=hwdichotomizer(disc);

% calculates loss
miss_classified = 1-sum(Y == pred,'all')/numel(pred);



