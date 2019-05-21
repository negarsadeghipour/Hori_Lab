clear all; close all; clc
test_iter = 10;
time = (0:400)';
num_patients = 200;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 0; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
max_k_gr_non = log(2)/18/30;% ;
min_k_gr_non = 1e-5;% ;
max_k_decay_non = log(2)/5/30;
min_k_decay_non = log(2)/24/30;
k_gr_non_rnd = (max_k_gr_non - min_k_gr_non).*rand(num_patients,1) + min_k_gr_non;%mean([0 1/18/30]);
k_decay_non_rnd = (max_k_decay_non - min_k_decay_non).*rand(num_patients,1) + min_k_decay_non;%mean([1/(24*30) 1/150]);

m_k_gr_non_rnd = mean(k_gr_non_rnd);
m_k_decay_non_rnd = mean(k_decay_non_rnd);

%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 0; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
max_k_gr_agg = log(2)/2/30;
min_k_gr_agg = log(2)/18/300;
max_k_decay_agg = log(2)/30;
min_k_decay_agg = log(2)/5/30;
k_gr_agg_rnd = (max_k_gr_agg - min_k_gr_agg).*rand(num_patients,1) + min_k_gr_agg;%linspace(1/18/30,1/2/30,50); %day-1
k_decay_agg_rnd = (max_k_decay_agg - min_k_decay_agg).*rand(num_patients,1) + min_k_decay_agg;%linspace(1/150,1/30,50); %day-1


noise_i = 0;

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);
t_onset = 200;



for j = 1:num_patients
    for i = 1:numel(time)
        if i < t_onset
            C_healthy(i,j) = Ch0_rnd(j);
            C_unhealthy(i,j) = Ca0_rnd(j);
        else
            C_healthy(i,j) = Ch0_rnd(j)*exp((k_gr_non_rnd(j)./k_decay_non_rnd(j))*(1-exp(-k_decay_non_rnd(j)*time(i-(t_onset-1)))));
            C_unhealthy(i,j) = Ca0_rnd(j)*exp((k_gr_agg_rnd(j)/k_decay_agg_rnd(j))*(1-exp(-k_decay_agg_rnd(j)*time(i-(t_onset-1)))));
        end
    end
end


C_healthy_noise1 = zeros(numel(time),num_patients);
C_unhealthy_noise1 = zeros(numel(time),num_patients);


for j = 1:num_patients
    C_healthy_noise1(:,j) = C_healthy(:,j)+noise_i*rand(1,numel(C_healthy(:,j)))'.*C_healthy(:,j)/100;
    C_unhealthy_noise1(:,j) = C_unhealthy(:,j)+noise_i*rand(1,numel(C_unhealthy(:,j)))'.*C_unhealthy(:,j)/100;
end


figure;plot(time,squeeze(C_healthy_noise1(:,10)),'r-',time,squeeze(C_unhealthy_noise1(:,10)),'b-','LineWidth', 2);%,time,squeeze(C_unhealthy_noise1(:,1,:)),'b-'); 
title('');
xlabel('Time (day)'); ylabel('Biomarker level (ng/mL)');
set(gca,'box','off','TickDir','out',...
    'FontSize',20,'FontName','Helvetica','LineWidth', 3);

% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
sample_interval = linspace(1,50,25);
observationspan = linspace(200,400,20);
tf_healthy = false(N,numel(sample_interval),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:) = true;

for m = 1:numel(sample_interval) 
    n = 1;
    meanF1(m) = 0;
    while ((meanF1(m) < .6) || isnan(meanF1(m))) && (n < numel(observationspan))
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
        meanF1(m) = mean(mean(f1Scores(:,m,:),1),3);
    end    
    observ_t(m) = observationspan(n);
       
end




figure;
h = plot(sample_interval, observ_t-200,'*-','LineWidth', 3);
title('f1 score - 5% noise');
xlabel('Days between samples (day)'); ylabel('Observation span-after cancer onset (day)');
set(gca,'box','off','TickDir','out',...
    'FontSize',20,'FontName','Helvetica','LineWidth', 3);