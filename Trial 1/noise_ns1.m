clear all; close all; clc
test_iter = 10;
time = (0:400)';
num_patients = 200;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
k_gr_non_rnd = normrnd(0,1/18/30,[num_patients,1]);%mean([0 1/18/30]);%linspace(1/18/30,1/60,50); %day-1
k_decay_non_rnd = normrnd(1/(24*30), 1/150,[num_patients,1]);%mean([1/(24*30) 1/150]);%linspace(1/30,1/150,50); %day-1

%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
k_gr_agg_rnd = normrnd(1/18/30,1/60,[num_patients,1]);%linspace(1/18/30,1/60,50); %day-1
k_decay_agg_rnd = normrnd(1/30,1/150,[num_patients,1]);%linspace(1/30,1/150,50); %day-1

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

% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
tf_healthy = false(N,numel(noise_i),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:) = true;

for l = 1:test_iter
    for n = 1:numel(noise_i)
            tf_healthy2(:,n,l) = tf_healthy(randperm(N),n,l);   % randomise order
            C_healthy_noise1_train(:,:,n) = C_healthy_noise1(:,squeeze(tf_healthy2(:,n,l)),n);
            C_healthy_noise1_test(:,:,n) = C_healthy_noise1(:,squeeze(~tf_healthy2(:,n,l)),n);
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_train(:,:,n) = C_unhealthy_noise1(:,squeeze(tf_healthy2(:,n,l)),n);
            C_unhealthy_noise1_test(:,:,n) = C_unhealthy_noise1(:,squeeze(~tf_healthy2(:,n,l)),n);
            
            train_data(:,:,n) = [C_healthy_noise1_train(:,:,n),C_unhealthy_noise1_train(:,:,n)];
            test_data(:,:,n) = [C_healthy_noise1_test(:,:,n),C_unhealthy_noise1_test(:,:,n)];
            train_labels(:,n) = [ones(1,p*N),zeros(1,p*N)];
            test_true_labels(:,n) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
            
            [predicted_labels(:,n),nn_index,accuracy(:,n)] = KNN_(3,train_data(:,:,n)',train_labels(:,n)',test_data(:,:,n)');
            [cm(:,:,n,l),gn(:,:,n,l)] = confusionmat(test_true_labels(:,n)',predicted_labels(:,n));
            % precision, recall and f1Scores
            precision(:,n,l) = diag(cm(:,:,n,l))./sum(cm(:,:,n,l),2);
            recall(:,n,l) = diag(cm(:,:,n,l))./sum(cm(:,:,n,l),1)';
            f1Scores(:,n,l) = 2*(precision(:,n,l).*recall(:,n,l))./(precision(:,n,l)+recall(:,n,l));
            
    end
end

meanF1 = mean(mean(f1Scores,1),3);


figure;
h = plot(noise_i, squeeze(meanF1),'*');
title('Noise');
xlabel('Noise (%)'); ylabel('f1 score');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
 set(h,'linestyle','none');