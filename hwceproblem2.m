%% Read the data 
clear; clc; close all;

abalone = readtable('abalone_dataset.txt');
X = abalone(:,1:8);
Y = table2array(abalone(:,9));

%% first 3 features 100 samples for training rest for test
X_train=X(1:100,1:3);
Y_train=Y(1:100);
X_test=X(101:end,1:3);
Y_test=Y(101:end);

% fit a naive bayes classifier
Mdl=fitcnb(X_train,Y_train);

% test the classifier using training data
resubstitution_loss = resubLoss(Mdl) % Training Loss
resub_Labels = resubPredict(Mdl); 
ConfusionMat1 = confusionmat(Y_train,resub_Labels);
figure
heatmap(ConfusionMat1)
set(gca,"XLabel","Predicted Class","YLabel","True Class")


%calculate loss using test data
test_loss=loss(Mdl,X_test,Y_test) % Test loss
test_labels=predict(Mdl,X_test);
ConfusionMat2 = confusionmat(Y_test,test_labels);
figure
heatmap(ConfusionMat2)
set(gca,"XLabel","Predicted Class","YLabel","True Class")

%% first 3 features 1000 samples for training rest for test
X_train=X(1:1000,1:3);
Y_train=Y(1:1000);
X_test=X(1001:end,1:3);
Y_test=Y(1001:end);

% fit a naive bayes classifier
Mdl=fitcnb(X_train,Y_train);

% test the classifier using training data
resubstitution_loss = resubLoss(Mdl) % Training Loss
resub_Labels = resubPredict(Mdl); 
ConfusionMat1 = confusionmat(Y_train,resub_Labels);
figure
heatmap(ConfusionMat1)
set(gca,"XLabel","Predicted Class","YLabel","True Class")


%calculate loss using test data
test_loss=loss(Mdl,X_test,Y_test) % Test loss
test_labels=predict(Mdl,X_test);
ConfusionMat2 = confusionmat(Y_test,test_labels);
figure
heatmap(ConfusionMat2)
set(gca,"XLabel","Predicted Class","YLabel","True Class")

%% first 3 features 2000 samples for training rest for test
X_train=X(1:2000,1:3);
Y_train=Y(1:2000);
X_test=X(2001:end,1:3);
Y_test=Y(2001:end);

% fit a naive bayes classifier
Mdl=fitcnb(X_train,Y_train);

% test the classifier using training data
resubstitution_loss = resubLoss(Mdl) % Training Loss
resub_Labels = resubPredict(Mdl); 
ConfusionMat1 = confusionmat(Y_train,resub_Labels);
figure
heatmap(ConfusionMat1)
set(gca,"XLabel","Predicted Class","YLabel","True Class")


%calculate loss using test data
test_loss=loss(Mdl,X_test,Y_test) % Test loss
test_labels=predict(Mdl,X_test);
ConfusionMat2 = confusionmat(Y_test,test_labels);
figure
heatmap(ConfusionMat2)
set(gca,"XLabel","Predicted Class","YLabel","True Class")

%% first 3 features 10 fold cross validation
X_train=X(:,1:3);
Y_train=Y;
% fit a naive bayes classifier
Mdl=fitcnb(X_train,Y_train);
CVSMdl=crossval(Mdl);

% test the classifier using training data
resubstitution_loss = kfoldLoss(CVSMdl)
resub_Labels = kfoldPredict(CVSMdl);
ConfusionMat1 = confusionmat(Y_train,resub_Labels);
figure
heatmap(ConfusionMat1)
set(gca,"XLabel","Predicted Class","YLabel","True Class")


%% all features 100 samples for training rest for test
X_train=X(1:100,:);
Y_train=Y(1:100);
X_test=X(101:end,:);
Y_test=Y(101:end);

% fit a naive bayes classifier
Mdl=fitcnb(X_train,Y_train);

% test the classifier using training data
resubstitution_loss = resubLoss(Mdl) % Training Loss
resub_Labels = resubPredict(Mdl); 
ConfusionMat1 = confusionmat(Y_train,resub_Labels);
figure
heatmap(ConfusionMat1)
set(gca,"XLabel","Predicted Class","YLabel","True Class")


%calculate loss using test data
test_loss=loss(Mdl,X_test,Y_test) % Test loss
test_labels=predict(Mdl,X_test);
ConfusionMat2 = confusionmat(Y_test,test_labels);
figure
heatmap(ConfusionMat2)
set(gca,"XLabel","Predicted Class","YLabel","True Class")

%% all features 1000 samples for training rest for test
X_train=X(1:1000,:);
Y_train=Y(1:1000);
X_test=X(1001:end,:);
Y_test=Y(1001:end);

% fit a naive bayes classifier
Mdl=fitcnb(X_train,Y_train);

% test the classifier using training data
resubstitution_loss = resubLoss(Mdl) % Training Loss
resub_Labels = resubPredict(Mdl); 
ConfusionMat1 = confusionmat(Y_train,resub_Labels);
figure
heatmap(ConfusionMat1)
set(gca,"XLabel","Predicted Class","YLabel","True Class")


%calculate loss using test data
test_loss=loss(Mdl,X_test,Y_test) % Test loss
test_labels=predict(Mdl,X_test);
ConfusionMat2 = confusionmat(Y_test,test_labels);
figure
heatmap(ConfusionMat2)
set(gca,"XLabel","Predicted Class","YLabel","True Class")

%% all features 2000 samples for training rest for test
X_train=X(1:2000,:);
Y_train=Y(1:2000);
X_test=X(2001:end,:);
Y_test=Y(2001:end);

% fit a naive bayes classifier
Mdl=fitcnb(X_train,Y_train);

% test the classifier using training data
resubstitution_loss = resubLoss(Mdl) % Training Loss 
resub_Labels = resubPredict(Mdl); 
ConfusionMat1 = confusionmat(Y_train,resub_Labels);
figure
heatmap(ConfusionMat1)
set(gca,"XLabel","Predicted Class","YLabel","True Class")


%calculate loss using test data
test_loss=loss(Mdl,X_test,Y_test) % Test loss % Test Loss
test_labels=predict(Mdl,X_test); % Predict test labels

ConfusionMat2 = confusionmat(Y_test,test_labels);
figure
heatmap(ConfusionMat2)
set(gca,"XLabel","Predicted Class","YLabel","True Class")

%% all features 10 fold cross validation
X_train=X;
Y_train=Y;
% fit a naive bayes classifier
Mdl=fitcnb(X_train,Y_train);
CVSMdl=crossval(Mdl);

% test the classifier using training data
resubstitution_loss = kfoldLoss(CVSMdl) % Training Loss
resub_Labels = kfoldPredict(CVSMdl);

ConfusionMat1 = confusionmat(Y_train,resub_Labels);
figure
heatmap(ConfusionMat1)
set(gca,"XLabel","Predicted Class","YLabel","True Class")
