clear all; close all; clc
test_iter = 11;
time = (0:400)';
num_patients = 200;
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
k_gr = 1/60;%linspace(1/18/30,1/60,50); %day
k_decay = 1/150;%linspace(1/30,1/150,50); %day


C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);

% C_healthy_noise1 = zeros(numel(time),num_patients);
% C_unhealthy_noise1 = zeros(numel(time),num_patients);
t_onset = 200;


for j = 1:num_patients
    for i = 1:numel(time)
        C_healthy(i,j) = Ch0_rnd(j);
        if i < t_onset
            C_unhealthy(i,j) = Ca0_rnd(j);
        else
            C_unhealthy(i,j) = Ca0_rnd(j)*exp((k_gr/k_decay)*(1-exp(-k_decay*time(i-(t_onset-1)))));
        end
    end
end



for k = 1:num_patients
    
    C_healthy_noise1(:,k) = C_healthy(:,k)+5*rand(1,numel(C_healthy(:,k)))'.*C_healthy(:,k)/100;
    C_unhealthy_noise1(:,k) = C_unhealthy(:,k)+5*rand(1,numel(C_unhealthy(:,k)))'.*C_unhealthy(:,k)/100;
    
end

% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
sample_interval = linspace(1,1,30);
observationspan = linspace(200,400,20);
tf_healthy = false(N,numel(sample_interval),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:) = true;
figure; plot(time(1:round(sample_interval(10)):round(observationspan(10))),C_healthy_noise1(1:round(sample_interval(10)):round(observationspan(10)),1),'r',time(1:round(sample_interval(10)):round(observationspan(10))),C_unhealthy_noise1(1:round(sample_interval(10)):round(observationspan(10)),1),'b');

% 
% meanF1 = zeros(numel(sample_interval),test_iter);
tf_healthy2 = false(size(tf_healthy));
C_healthy_noise1_train = zeros(numel(time),round(p*N), numel(sample_interval),test_iter);
C_healthy_noise1_test = zeros(numel(time),round((1-p)*N), numel(sample_interval),test_iter);
C_unhealthy_noise1_train = zeros(numel(time),p*N, numel(sample_interval),test_iter);
C_unhealthy_noise1_test = zeros(numel(time),round((1-p)*N), numel(sample_interval),test_iter);
train_labels = zeros(2*p*N, numel(sample_interval),test_iter);
test_true_labels = zeros(2*p*N, numel(sample_interval),test_iter);

for l = 1:1%test_iter
    for m = 1:1%numel(sample_interval)
        n = 1;  
        meanF1 = 0;   
        clear train_data test_data
        while meanF1 < 0.8
            tf_healthy2(:,m,l) = tf_healthy(randperm(N),m,l);   % randomise order
            C_healthy_noise1_train(:,:,m,l) = C_healthy_noise1(:,squeeze(tf_healthy2(:,m,l)));
            C_healthy_noise1_test(:,:,m,l) = C_healthy_noise1(:,squeeze(~tf_healthy2(:,m,l)));
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_train(:,:,m,l) = C_unhealthy_noise1(:,squeeze(tf_healthy2(:,m,l)));
            C_unhealthy_noise1_test(:,:,m,l) = C_unhealthy_noise1(:,squeeze(~tf_healthy2(:,m,l)));
            
            train_data = [C_healthy_noise1_train(1:round(sample_interval(m)):round(observationspan(n)),:,m,l),C_unhealthy_noise1_train(1:round(sample_interval(m)):round(observationspan(n)),:,m,l)];
            test_data = [C_healthy_noise1_test(1:round(sample_interval(m)):round(observationspan(n)),:,m,l),C_unhealthy_noise1_test(1:round(sample_interval(m)):round(observationspan(n)),:,m,l)];
            train_labels(:,m,l) = [ones(1,p*N),zeros(1,p*N)];
            test_true_labels(:,m,l) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
            
            [predicted_labels(:,m,l),nn_index,accuracy(:,m,l)] = KNN_(3,train_data',train_labels(:,m,l)',test_data');
            [cm(:,:,m,l),gn(:,:,m,l)] = confusionmat(test_true_labels(:,m,l)',predicted_labels(:,m,l));
            % precision, recall and f1Scores
            %             accuracy1(:,m,n,l) =  sum(diag(cm(:,:,n,l)))./sum(sum(cm(:,:,n,l)));
            precision(:,m,l) = diag(cm(:,:,m,l))./sum(cm(:,:,m,l),2);
            recall(:,m,l) = diag(cm(:,:,m,l))./sum(cm(:,:,m,l),1)';
            f1Scores(:,m,l) = 2*(precision(:,m,l).*recall(:,m,l))./(precision(:,m,l)+recall(:,m,l));
            n = n+1;
            meanF1 = squeeze(mean(mean(f1Scores,4),1));
        end
        
        observ_t(m,l) = observationspan(n);
        
    end
end




figure;
h = scatter(sample_interval, observ_t);
title('f1 score - 5% noise - Aggressive');
xlabel('Days between samples (day)'); ylabel('Observation span (day)');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);