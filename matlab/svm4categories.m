clear all
close all
clc

load('./data/Training_Data.mat')
load('./data/Test_Data.mat')

load('./data4/train_index.mat')
train_index = Index;
load('./data4/test_index.mat')
test_index = Index;

classes = [6, 7, 8]

if length(classes)==2
% Train the SVM Classifier
c1 = fitcsvm(candidates,train_index','KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[6, 7, 8]);
else
    svm = templateSVM('KernelFunction','rbf', 'BoxConstraint',Inf);
    c1 = fitcecoc(candidates, train_index', 'Learners', svm)
end

[labels, ~] = predict(c1, test_candidates);
[labels_t, ~] = predict(c1, candidates);

plot(test_index, "^"); hold on
plot(labels, "rX");
legend(["Actual order for test samples", "SVM outputs for test samples"]);
ylabel("Order")
xlabel("Testing samples")
xlim([-1 37])
% title("Error:"+num2str(1-sum(test_index'==labels)/length(labels)))
ylim([5.5 9])

figure
plot(train_index, "s"); hold on
plot(labels_t, "r+");
legend(["Actual order for train samples", "SVM outputs for train samples"]);
ylabel("Order")
xlabel("Training samples")
xlim([-1 65])
% title("Error:"+num2str(1-sum(train_index'==labels_t)/length(labels_t)))
ylim([5.5 9])