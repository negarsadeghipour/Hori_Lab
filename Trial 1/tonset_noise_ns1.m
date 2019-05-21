%% This code generates:
% a. 
%% 
clear all; close all; clc
time = (0:400)';
num_patients = 200;
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
k_gr = 1e-2; 
k_decay = 1e-3;

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);

% C_healthy_noise1 = zeros(numel(time),num_patients);
% C_unhealthy_noise1 = zeros(numel(time),num_patients);
t_onset_i = randi(200,num_patients,1);

for j = 1:num_patients
    for i = 1:numel(time)
        C_healthy(i,j) = Ch0_rnd(j);
        if i < t_onset_i(j)
            C_unhealthy(i,j) = Ca0_rnd(j);
        else
            C_unhealthy(i,j) = Ca0_rnd(j)*exp((k_gr/k_decay)*(1-exp(-k_decay*time(i-(t_onset_i(j)-1)))));
        end
    end
end

for k = 1:num_patients
    
    C_healthy_noise1(:,k) = C_healthy(:,k)+5*rand(1,numel(C_healthy(:,k)))'.*C_healthy(:,k)/100;
    C_unhealthy_noise1(:,k) = C_unhealthy(:,k)+5*rand(1,numel(C_unhealthy(:,k)))'.*C_unhealthy(:,k)/100;
    
end
figure;
plot(time, C_healthy_noise1, 'r', time, C_unhealthy_noise1, 'b');
title('Different onset time');
xlabel('time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',18,'FontName','Helvetica','YScale','log','LineWidth', 3);
%% KNN
% model 1

p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows 
tf_healthy = false(N,1);    % create logical index vector
tf_healthy(1:round(p*N)) = true;     
tf_healthy = tf_healthy(randperm(N));   % randomise order
C_healthy_noise1_train = C_healthy_noise1(:,tf_healthy); 
C_healthy_noise1_test = C_healthy_noise1(:,~tf_healthy);
% for now using the same index of random selection for healthy and
% unhealthy
C_unhealthy_noise1_train = C_unhealthy_noise1(:,tf_healthy); 
C_unhealthy_noise1_test = C_unhealthy_noise1(:,~tf_healthy);
train_data = [C_healthy_noise1_train,C_unhealthy_noise1_train];
test_data = [C_healthy_noise1_test,C_unhealthy_noise1_test];
train_labels = [ones(1,p*N),zeros(1,p*N)];
test_true_labels = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];

[predicted_labels,nn_index,accuracy] = KNN_(3,train_data',train_labels',test_data');
[cm,gn] = confusionmat(test_true_labels',predicted_labels);
% precision, recall and f1Scores
precision = diag(cm)./sum(cm,2);

recall = diag(cm)./sum(cm,1)';

f1Scores = 2*(precision.*recall)./(precision+recall);

meanF1 = mean(f1Scores);