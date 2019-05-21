clear all; close all; clc
test_iter = 10;
time = (0:400)';
num_patients = 200;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
max_k_gr_non = 1/18/30;
min_k_gr_non = 0;
max_k_decay_non = 1/5/30;
min_k_decay_non = 1/24/30;
k_gr_non_rnd = (max_k_gr_non - min_k_gr_non).*rand(num_patients,1) + min_k_gr_non;%mean([0 1/18/30]);
k_decay_non_rnd = (max_k_decay_non - min_k_decay_non).*rand(num_patients,1) + min_k_decay_non;%mean([1/(24*30) 1/150]);

m_k_gr_non_rnd = mean(k_gr_non_rnd);
sd_k_gr_non_rnd = std(k_gr_non_rnd);

m_k_decay_non_rnd = mean(k_decay_non_rnd);
sd_k_decay_non_rnd = std(k_decay_non_rnd);

%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
max_k_gr_agg = 1/2/30;
min_k_gr_agg = 1/18/30;
max_k_decay_agg = 1/30;
min_k_decay_agg = 1/5/30;
k_gr_agg_rnd = (max_k_gr_agg - min_k_gr_agg).*rand(num_patients,1) + min_k_gr_agg;%linspace(1/18/30,1/2/30,50); %day-1
k_decay_agg_rnd = (max_k_decay_agg - min_k_decay_agg).*rand(num_patients,1) + min_k_decay_agg;%linspace(1/150,1/30,50); %day-1

m_k_gr_agg_rnd = mean(k_gr_agg_rnd);
sd_k_gr_agg_rnd = std(k_gr_agg_rnd);

m_k_decay_agg_rnd = mean(k_decay_agg_rnd);
sd_k_decay_agg_rnd = std(k_decay_agg_rnd);


noise_i = 0:5:90;

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

C_healthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));
C_unhealthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));

for n = 1:numel(noise_i)
    for k = 1:num_patients
        
        C_healthy_noise1(:,k,n) = C_healthy(:,k)+noise_i(n)*rand(1,numel(C_healthy(:,k)))'.*C_healthy(:,k)/100;
        C_unhealthy_noise1(:,k,n) = C_unhealthy(:,k)+noise_i(n)*rand(1,numel(C_unhealthy(:,k)))'.*C_unhealthy(:,k)/100;
        
    end
end
figure;plot(time,squeeze(C_healthy_noise1(:,1:5,5)),'r-',time,squeeze(C_unhealthy_noise1(:,1:5,5)),'b-'); 
% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
tf_healthy = false(N,numel(noise_i),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:) = true;


for n = 1:numel(noise_i)
    for l = 1:test_iter
        tf_healthy2(:,n,l) = tf_healthy(randperm(N),n,l);   % randomise order
        C_healthy_noise1_train(:,:,n,l) = C_healthy_noise1(:,squeeze(tf_healthy2(:,n,l)),n);
        C_healthy_noise1_test(:,:,n,l) = C_healthy_noise1(:,squeeze(~tf_healthy2(:,n,l)),n);
        % for now using the same index of random selection for healthy and
        % unhealthy
        C_unhealthy_noise1_train(:,:,n,l) = C_unhealthy_noise1(:,squeeze(tf_healthy2(:,n,l)),n);
        C_unhealthy_noise1_test(:,:,n,l) = C_unhealthy_noise1(:,squeeze(~tf_healthy2(:,n,l)),n);
        
        train_data(:,:,n,l) = [C_healthy_noise1_train(:,:,n,l),C_unhealthy_noise1_train(:,:,n,l)];
        test_data(:,:,n,l) = [C_healthy_noise1_test(:,:,n,l),C_unhealthy_noise1_test(:,:,n,l)];
        train_labels(:,n,l) = [ones(1,p*N),zeros(1,p*N)];
        test_true_labels(:,n,l) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
        
        [predicted_labels(:,n,l),nn_index,accuracy] = KNN_(3,train_data(:,:,n,l)',train_labels(:,n,l)',test_data(:,:,n,l)');
        [cm(:,:,n,l),gn(:,:,n,l)] = confusionmat(test_true_labels(:,n,l)',predicted_labels(:,n,l));
        % precision, recall and f1Scores
        precision(:,n,l) = diag(cm(:,:,n,l))./sum(cm(:,:,n,l),2);
        recall(:,n,l) = diag(cm(:,:,n,l))./sum(cm(:,:,n,l),1)';
        f1Scores(:,n,l) = 2*(precision(:,n,l).*recall(:,n,l))./(precision(:,n,l)+recall(:,n,l));
    end
    meanF1(n) = mean(mean(f1Scores(:,n,:)));
end




figure;
h = plot(noise_i, squeeze(meanF1),'*','MarkerSize',12);
title('Noise');
xlabel('Noise (%)'); ylabel('f1 score');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
set(h,'linestyle','none');