% total observation time vs sampling time to reach certain f1 score
%%
clear all; close all; clc
test_iter = 10;
time = (0:400)';
num_patients = 200;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
k_gr_non_rnd = normrnd(0, 1/18/30,[num_patients,1]);%mean([0 1/18/30]);
k_decay_non_rnd = normrnd(1/(24*30), 1/150,[num_patients,1]);%mean([1/(24*30) 1/150]);

%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
k_gr_agg_rnd = normrnd(1/18/30,1/60,[num_patients,1]);%linspace(1/18/30,1/60,50); %day-1
k_decay_agg_rnd = normrnd(1/30,1/150,[num_patients,1]);%linspace(1/30,1/150,50); %day-1


C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);
t_onset = 200;

for j = 1:num_patients
    for i = 1:numel(time)
        if i < t_onset
            C_healthy(i,j) = Ch0_rnd(j);
            C_unhealthy(i,j) = Ca0_rnd(j);
        else
            C_healthy(i,j) = Ch0_rnd(j)*exp((k_gr_non_rnd(j)/k_decay_non_rnd(j))*(1-exp(-k_decay_non_rnd(j)*time(i-(t_onset-1)))));
            C_unhealthy(i,j) = Ca0_rnd(j)*exp((k_gr_agg_rnd(j)/k_decay_agg_rnd(j))*(1-exp(-k_decay_agg_rnd(j)*time(i-(t_onset-1)))));
        end
    end
end

C_healthy_noise1 = zeros(numel(time),num_patients);
C_unhealthy_noise1 = zeros(numel(time),num_patients);


for k = 1:num_patients
    
    C_healthy_noise1(:,k) = C_healthy(:,k)+5*rand(1,numel(C_healthy(:,k)))'.*C_healthy(:,k)/100;
    C_unhealthy_noise1(:,k) = C_unhealthy(:,k)+5*rand(1,numel(C_unhealthy(:,k)))'.*C_unhealthy(:,k)/100;
    
end



% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
sample_interval = round(linspace(1,50,20));
observationspan = round(linspace(200,400,20));
tf_healthy = false(N,numel(sample_interval),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:) = true;
% figure; plot(time(1:round(sample_interval(10)):round(observationspan(10))),C_healthy_noise1(1:round(sample_interval(10)):round(observationspan(10)),1),'r',time(1:round(sample_interval(10)):round(observationspan(10))),C_unhealthy_noise1(1:round(sample_interval(10)):round(observationspan(10)),1),'b');

%


for m = 1:numel(sample_interval)
    n = 1;
    meanF1(m) = 0;
    while (meanF1(m) < .70) && (n < numel(observationspan))
        for l = 1:test_iter
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
            
            [predicted_labels(:,m,l),nn_index,accuracy] = KNN_(3,train_data',train_labels(:,m,l)',test_data');
            [cm(:,:,m,l),gn(:,:,m,l)] = confusionmat(test_true_labels(:,m,l)',predicted_labels(:,m,l));
            % precision, recall and f1Scores
            %             accuracy1(:,m,n,l) =  sum(diag(cm(:,:,n,l)))./sum(sum(cm(:,:,n,l)));
            precision(:,m,l) = diag(cm(:,:,m,l))./sum(cm(:,:,m,l),2);
            recall(:,m,l) = diag(cm(:,:,m,l))./sum(cm(:,:,m,l),1)';
            f1Scores(:,m,l) = 2*(precision(:,m,l).*recall(:,m,l))./(precision(:,m,l)+recall(:,m,l));
            
        end
        n = n+1;
        x(m) = n;
        meanF1(m) = mean(mean(f1Scores(:,m,:)));
    end
    observ_t(m) = observationspan(n);
    
end




figure;
h = plot(sample_interval, observ_t-200,'*-','LineWidth', 3);
title('f1 score - 5% noise');
xlabel('Days between samples (day)'); ylabel('Observation span-after cancer onset (day)');
set(gca,'box','off','TickDir','out',...
    'FontSize',20,'FontName','Helvetica','LineWidth', 3);