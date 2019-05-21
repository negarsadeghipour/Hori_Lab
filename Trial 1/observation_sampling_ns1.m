clear all; close all; clc
test_iter = 10;
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
sample_interval = linspace(1,40,10);
observationspan = linspace(200,400,20);

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);

% C_healthy_noise1 = zeros(numel(time),num_patients);
% C_unhealthy_noise1 = zeros(numel(time),num_patients);
t_onset = 200;

for m = 1:numel(observationspan)
    for k = 1:numel(sample_interval)
        for j = 1:num_patients
            for i = 1:numel(time)
                C_healthy(i,j,k,m) = Ch0_rnd(j);
                if i < t_onset
                    C_unhealthy(i,j,k,m) = Ca0_rnd(j);
                else
                    C_unhealthy(i,j,k,m) = Ca0_rnd(j)*exp((k_gr/k_decay)*(1-exp(-k_decay*time(i-(t_onset-1)))));
                end
            end
        end
    end
end

for n = 1:numel(observationspan)
    for l = 1:numel(sample_interval)
        for k = 1:num_patients
            
            C_healthy_noise1(:,k,l,n) = C_healthy(:,k,l,n)+5*rand(1,numel(C_healthy(:,k,l,n)))'.*C_healthy(:,k,l,n)/100;
            C_unhealthy_noise1(:,k,l,n) = C_unhealthy(:,k,l,n)+5*rand(1,numel(C_unhealthy(:,k,l,n)))'.*C_unhealthy(:,k,l,n)/100;
            
        end
    end
end

% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
tf_healthy = false(N,numel(sample_interval),numel(observationspan),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:,:) = true;

for l = 1:test_iter
    for n = 1:numel(observationspan)
        for m = 1:numel(sample_interval)
            tf_healthy(:,m,n) = tf_healthy(randperm(N),m,n);   % randomise order
            C_healthy_noise1_train(:,:,m,n) = C_healthy_noise1(:,squeeze(tf_healthy(:,m,n)),m,n);
            C_healthy_noise1_test(:,:,m,n) = C_healthy_noise1(:,squeeze(~tf_healthy(:,m,n)),m,n);
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_train(:,:,m,n) = C_unhealthy_noise1(:,squeeze(tf_healthy(:,m,n)),m,n);
            C_unhealthy_noise1_test(:,:,m,n) = C_unhealthy_noise1(:,squeeze(~tf_healthy(:,m,n)),m,n);
            
            train_data = [C_healthy_noise1_train(1:round(sample_interval(m)):round(observationspan(n)),:,m,n),C_unhealthy_noise1_train(1:round(sample_interval(m)):round(observationspan(n)),:,m,n)];
            test_data = [C_healthy_noise1_test(1:round(sample_interval(m)):round(observationspan(n)),:,m,n),C_unhealthy_noise1_test(1:round(sample_interval(m)):round(observationspan(n)),:,m,n)];
            train_labels(:,m,n) = [ones(1,p*N),zeros(1,p*N)];
            test_true_labels(:,m,n) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
            
            [predicted_labels(:,m,n),nn_index,accuracy(:,m,n)] = KNN_(3,train_data',train_labels(:,m,n)',test_data');
            [cm(:,:,m,n,l),gn(:,:,m,n,l)] = confusionmat(test_true_labels(:,m,n)',predicted_labels(:,m,n));
            % precision, recall and f1Scores
%             accuracy1(:,m,n,l) =  sum(diag(cm(:,:,n,l)))./sum(sum(cm(:,:,n,l)));
            precision(:,m,n,l) = diag(cm(:,:,m,n,l))./sum(cm(:,:,m,n,l),2);
            recall(:,m,n,l) = diag(cm(:,:,m,n,l))./sum(cm(:,:,m,n,l),1)';
            f1Scores(:,m,n,l) = 2*(precision(:,m,n,l).*recall(:,m,n,l))./(precision(:,m,n,l)+recall(:,m,n,l));
            
        end
    end
end

meanF1 = mean(mean(f1Scores,4),1);


figure;
h = surf(observationspan, sample_interval, squeeze(meanF1));
title('f1 score - 5% noise - Aggressive');
xlabel('Observation span (day)'); ylabel('Days between samples (day)');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
 set(h,'linestyle','none');