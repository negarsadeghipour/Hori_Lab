%% This code generates:
% a. classification between aggressive and no cancer patients (TD = 60 day)
% b. classification between non-aggressive and no cancer patients (TD = 18 month)
% c. classification in a range of k_decay and k_growth
%% a. Aggressive and non-aggressive distribution of kdecay and kgrowth
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
% k_gr_agg_rnd = normrnd(1/18/30,1/60,[num_patients,1]);%linspace(1/18/30,1/60,50); %day-1
% k_decay_agg_rnd = normrnd(1/30,1/150,[num_patients,1]);%linspace(1/30,1/150,50); %day-1
% 
k_gr_agg_rnd = normrnd(1/18/30,1/60,[num_patients,1]);
k_decay_agg = linspace(1/30,1/150,10);%linspace(1/30,1/150,50); %day-1
k_decay_agg_std = 0; %day-1
for i = 1:numel(k_decay_agg)
k_decay_agg_rnd(:,i) = k_decay_agg_std.*randn(num_patients,1) + k_decay_agg(i);
end

%

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

C_healthy_noise1 = zeros(numel(time),num_patients,numel(k_decay_agg));
C_unhealthy_noise1 = zeros(numel(time),num_patients,numel(k_decay_agg));

for n = 1:numel(k_decay_agg)
    for k = 1:num_patients
        
        C_healthy_noise1(:,k,n) = C_healthy(:,k)+5*rand(1,numel(C_healthy(:,k)))'.*C_healthy(:,k)/100;
        C_unhealthy_noise1(:,k,n) = C_unhealthy(:,k)+5*rand(1,numel(C_unhealthy(:,k)))'.*C_unhealthy(:,k)/100;
        
    end
end

figure;
plot(time, squeeze(C_healthy_noise1(:,1,1)), 'r', time, squeeze(C_unhealthy_noise1(:,1,:)), 'b');
title({'Aggressive','K_{decay} range'});
xlabel('time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',18,'FontName','Helvetica','YScale','log','LineWidth', 3);
% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
tf_healthy = false(N,numel(k_decay_agg),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:) = true;

for l = 1:test_iter
    for n = 1:numel(k_decay_agg)
            tf_healthy(:,n) = tf_healthy(randperm(N),n);   % randomise order
            C_healthy_noise1_train(:,:,n) = C_healthy_noise1(:,squeeze(tf_healthy(:,n)),n);
            C_healthy_noise1_test(:,:,n) = C_healthy_noise1(:,squeeze(~tf_healthy(:,n)),n);
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_train(:,:,n) = C_unhealthy_noise1(:,squeeze(tf_healthy(:,n)),n);
            C_unhealthy_noise1_test(:,:,n) = C_unhealthy_noise1(:,squeeze(~tf_healthy(:,n)),n);
            
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
h = plot(k_decay_agg, squeeze(meanF1),'*');
title('kdecay');
xlabel('k decay agg (day^{-1})'); ylabel('f1 score');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
 set(h,'linestyle','none');

%% c. different ranges of k_decay and k_growth of aggressive cancer
clear all; close all; clc

t_onset = 200;

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

k_gr_agg = linspace(1/18/30,1/6,10);%linspace(1/18/30,1/60,50); %day-1
k_gr_agg_std = 0.01; %day-1
k_decay_agg = linspace(1/30,1/150,10);%linspace(1/30,1/150,50); %day-1
k_decay_agg_std = 0.01; %day-1
for i = 1:numel(k_gr_agg)
k_gr_agg_rnd(:,i) = k_gr_agg_std.*randn(num_patients,1) + k_gr_agg(i);
k_decay_agg_rnd(:,i) = k_decay_agg_std.*randn(num_patients,1) + k_decay_agg(i);
end


%


for m = 1:numel(k_gr_agg)
    for k = 1:numel(k_decay_agg)
        for j = 1:num_patients
            for i = 1:numel(time)
                C_healthy(i,j,k,m) = Ch0_rnd(j);
                if i < t_onset
                    C_healthy(i,j,k,m) = Ch0_rnd(j);
                    C_unhealthy(i,j,k,m) = Ca0_rnd(j);
                else
                    C_healthy(i,j,k,m) = Ch0_rnd(j)*exp((k_gr_non_rnd(j)/k_decay_non_rnd(j))*(1-exp(-k_decay_non_rnd(j)*time(i-(t_onset-1)))));
                    C_unhealthy(i,j,k,m) = Ca0_rnd(j)*exp((k_gr_agg_rnd(j)/k_decay_agg_rnd(j))*(1-exp(-k_decay_agg_rnd(j)*time(i-(t_onset-1)))));
                end
            end
        end
    end
end

for n = 1:numel(k_gr_agg)
    for l = 1:numel(k_decay_agg)
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
tf_healthy = false(N,numel(k_decay_agg),numel(k_gr_agg),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:,:) = true;

for l = 1:test_iter
    for n = 1:numel(k_gr_agg)
        for m = 1:numel(k_decay_agg)
            tf_healthy(:,m,n) = tf_healthy(randperm(N),m,n);   % randomise order
            C_healthy_noise1_train(:,:,m,n) = C_healthy_noise1(:,squeeze(tf_healthy(:,m,n)),m,n);
            C_healthy_noise1_test(:,:,m,n) = C_healthy_noise1(:,squeeze(~tf_healthy(:,m,n)),m,n);
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_train(:,:,m,n) = C_unhealthy_noise1(:,squeeze(tf_healthy(:,m,n)),m,n);
            C_unhealthy_noise1_test(:,:,m,n) = C_unhealthy_noise1(:,squeeze(~tf_healthy(:,m,n)),m,n);
            
            train_data(:,:,m,n) = [C_healthy_noise1_train(:,:,m,n),C_unhealthy_noise1_train(:,:,m,n)];
            test_data(:,:,m,n) = [C_healthy_noise1_test(:,:,m,n),C_unhealthy_noise1_test(:,:,m,n)];
            train_labels(:,m,n) = [ones(1,p*N),zeros(1,p*N)];
            test_true_labels(:,m,n) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
            
            [predicted_labels(:,m,n),nn_index,accuracy(:,m,n)] = KNN_(3,train_data(:,:,m,n)',train_labels(:,m,n)',test_data(:,:,m,n)');
            [cm(:,:,m,n,l),gn(:,:,m,n,l)] = confusionmat(test_true_labels(:,m,n)',predicted_labels(:,m,n));
            % precision, recall and f1Scores
            precision(:,m,n,l) = diag(cm(:,:,m,n,l))./sum(cm(:,:,m,n,l),2);
            recall(:,m,n,l) = diag(cm(:,:,m,n,l))./sum(cm(:,:,m,n,l),1)';
            f1Scores(:,m,n,l) = 2*(precision(:,m,n,l).*recall(:,m,n,l))./(precision(:,m,n,l)+recall(:,m,n,l));
            
        end
    end
end

meanF1 = mean(mean(f1Scores,4),1);


figure;
h = surf(k_gr_agg, k_decay_agg, squeeze(meanF1));
title('5% noise');
xlabel('k_{growth} (day^{-1})'); ylabel('k_{decay} (day^{-1})');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
 set(h,'linestyle','none');
%% d. different ranges of k_decay and k_growth of aggressive cancer
clear all; close all; clc

t_onset = 200;

test_iter = 10;
time = (0:400)';
num_patients = 200;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
k_gr_non = linspace(0,1/18/300,10);%linspace(1/18/30,1/60,50); %day-1
k_gr_non_std = 0; %day-1
k_decay_non = linspace(1/(24*30), 1/150,10);%linspace(1/30,1/150,50); %day-1
k_decay_non_std = 0; %day-1
for i = 1:numel(k_gr_non)
k_gr_non_rnd(:,i) = k_gr_non_std.*randn(num_patients,1) + k_gr_non(i);
k_decay_non_rnd(:,i) = k_decay_non_std.*randn(num_patients,1) + k_decay_non(i);
end

%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
k_gr_agg_rnd = normrnd(1/18/30,1/6,[num_patients,1]);%mean([0 1/18/30]);%linspace(1/18/30,1/60,50); %day-1
k_decay_agg_rnd = normrnd(1/30,1/150,[num_patients,1]);%mean([1/(24*30) 1/150]);%linspace(1/30,1/150,50); %day-1


%


for m = 1:numel(k_gr_non)
    for k = 1:numel(k_decay_non)
        for j = 1:num_patients
            for i = 1:numel(time)
                C_healthy(i,j,k,m) = Ch0_rnd(j);
                if i < t_onset
                    C_healthy(i,j,k,m) = Ch0_rnd(j);
                    C_unhealthy(i,j,k,m) = Ca0_rnd(j);
                else
                    C_healthy(i,j,k,m) = Ch0_rnd(j)*exp((k_gr_non_rnd(j)/k_decay_non_rnd(j))*(1-exp(-k_decay_non_rnd(j)*time(i-(t_onset-1)))));
                    C_unhealthy(i,j,k,m) = Ca0_rnd(j)*exp((k_gr_agg_rnd(j)/k_decay_agg_rnd(j))*(1-exp(-k_decay_agg_rnd(j)*time(i-(t_onset-1)))));
                end
            end
        end
    end
end

for n = 1:numel(k_gr_non)
    for l = 1:numel(k_decay_non)
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
tf_healthy = false(N,numel(k_decay_non),numel(k_gr_non),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:,:) = true;

for l = 1:test_iter
    for n = 1:numel(k_gr_non)
        for m = 1:numel(k_decay_non)
            tf_healthy(:,m,n) = tf_healthy(randperm(N),m,n);   % randomise order
            C_healthy_noise1_train(:,:,m,n) = C_healthy_noise1(:,squeeze(tf_healthy(:,m,n)),m,n);
            C_healthy_noise1_test(:,:,m,n) = C_healthy_noise1(:,squeeze(~tf_healthy(:,m,n)),m,n);
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_train(:,:,m,n) = C_unhealthy_noise1(:,squeeze(tf_healthy(:,m,n)),m,n);
            C_unhealthy_noise1_test(:,:,m,n) = C_unhealthy_noise1(:,squeeze(~tf_healthy(:,m,n)),m,n);
            
            train_data(:,:,m,n) = [C_healthy_noise1_train(:,:,m,n),C_unhealthy_noise1_train(:,:,m,n)];
            test_data(:,:,m,n) = [C_healthy_noise1_test(:,:,m,n),C_unhealthy_noise1_test(:,:,m,n)];
            train_labels(:,m,n) = [ones(1,p*N),zeros(1,p*N)];
            test_true_labels(:,m,n) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
            
            [predicted_labels(:,m,n),nn_index,accuracy(:,m,n)] = KNN_(3,train_data(:,:,m,n)',train_labels(:,m,n)',test_data(:,:,m,n)');
            [cm(:,:,m,n,l),gn(:,:,m,n,l)] = confusionmat(test_true_labels(:,m,n)',predicted_labels(:,m,n));
            % precision, recall and f1Scores
            precision(:,m,n,l) = diag(cm(:,:,m,n,l))./sum(cm(:,:,m,n,l),2);
            recall(:,m,n,l) = diag(cm(:,:,m,n,l))./sum(cm(:,:,m,n,l),1)';
            f1Scores(:,m,n,l) = 2*(precision(:,m,n,l).*recall(:,m,n,l))./(precision(:,m,n,l)+recall(:,m,n,l));
            
        end
    end
end

meanF1 = mean(mean(f1Scores,4),1);


figure;
h = surf(k_gr_non, k_decay_non, squeeze(meanF1));
title('5% noise');
xlabel('k_{growth} (day^{-1})'); ylabel('k_{decay} (day^{-1})');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
 set(h,'linestyle','none');
 %% d. different ranges of k_decay and k_growth with normalization
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
k_gr = linspace(1/18/30,1/60,50); %day
k_decay = linspace(1/30,1/150,50); %day

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);

% C_healthy_noise1 = zeros(numel(time),num_patients);
% C_unhealthy_noise1 = zeros(numel(time),num_patients);
t_onset = 200;

for m = 1:numel(k_gr)
    for k = 1:numel(k_decay)
        for j = 1:num_patients
            for i = 1:numel(time)
                C_healthy(i,j,k,m) = Ch0_rnd(j);
                if i < t_onset
                    C_unhealthy(i,j,k,m) = Ca0_rnd(j);
                else
                    C_unhealthy(i,j,k,m) = Ca0_rnd(j)*exp((k_gr(m)/k_decay(k))*(1-exp(-k_decay(k)*time(i-(t_onset-1)))));
                end
            end
        end
    end
end

for n = 1:numel(k_gr)
    for l = 1:numel(k_decay)
        for k = 1:num_patients
            
            C_healthy_noise1(:,k,l,n) = C_healthy(:,k,l,n)+50*rand(1,numel(C_healthy(:,k,l,n)))'.*C_healthy(:,k,l,n)/100;
            C_unhealthy_noise1(:,k,l,n) = C_unhealthy(:,k,l,n)+50*rand(1,numel(C_unhealthy(:,k,l,n)))'.*C_unhealthy(:,k,l,n)/100;
            
        end
    end
end

% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
tf_healthy = false(N,numel(k_decay),numel(k_gr),test_iter);    % create logical index vector
tf_healthy(1:round(p*N),:,:,:) = true;

for l = 1:test_iter
    for n = 1:numel(k_gr)
        for m = 1:numel(k_decay)
            tf_healthy(:,m,n) = tf_healthy(randperm(N),m,n);   % randomise order
            C_healthy_noise1_train(:,:,m,n) = C_healthy_noise1(:,squeeze(tf_healthy(:,m,n)),m,n);
            C_healthy_noise1_test(:,:,m,n) = C_healthy_noise1(:,squeeze(~tf_healthy(:,m,n)),m,n);
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_train(:,:,m,n) = C_unhealthy_noise1(:,squeeze(tf_healthy(:,m,n)),m,n);
            C_unhealthy_noise1_test(:,:,m,n) = C_unhealthy_noise1(:,squeeze(~tf_healthy(:,m,n)),m,n);
            
            train_data(:,:,m,n) = [C_healthy_noise1_train(:,:,m,n),C_unhealthy_noise1_train(:,:,m,n)];
            test_data(:,:,m,n) = [C_healthy_noise1_test(:,:,m,n),C_unhealthy_noise1_test(:,:,m,n)];
            train_labels(:,m,n) = [ones(1,p*N),zeros(1,p*N)];
            test_true_labels(:,m,n) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
            
            [predicted_labels(:,m,n),nn_index,accuracy(:,m,n)] = KNN_(3,train_data(:,:,m,n)',train_labels(:,m,n)',test_data(:,:,m,n)');
            [cm(:,:,m,n,l),gn(:,:,m,n,l)] = confusionmat(test_true_labels(:,m,n)',predicted_labels(:,m,n));
            % precision, recall and f1Scores
            precision(:,m,n,l) = diag(cm(:,:,m,n,l))./sum(cm(:,:,m,n,l),2);
            recall(:,m,n,l) = diag(cm(:,:,m,n,l))./sum(cm(:,:,m,n,l),1)';
            f1Scores(:,m,n,l) = 2*(precision(:,m,n,l).*recall(:,m,n,l))./(precision(:,m,n,l)+recall(:,m,n,l));
            
        end
    end
end

meanF1 = mean(mean(f1Scores,4),1);


figure;
h = surf(k_gr, k_decay, squeeze(meanF1));
title('5% noise');
xlabel('k_{growth} (day^{-1})'); ylabel('k_{decay} (day^{-1})');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
 set(h,'linestyle','none');

% normalization
for n = 1:numel(k_gr)
    for l = 1:numel(k_decay)
            
            mu100(l,n) = mean(mean(C_healthy_noise1(1:100,:,l,n),1));
            C_healthy_noise1_norm1(:,:,l,n) = C_healthy_noise1(:,:,l,n) - mu100(l,n);
            C_unhealthy_noise1_norm1(:,:,l,n) = C_unhealthy_noise1(:,:,l,n) - mu100(l,n);
            
    end
end

for n = 1:numel(k_gr)
    for l = 1:numel(k_decay)
            
            
            std100(l,n) = mean(std(C_healthy_noise1(1:100,:,l,n)));
            C_healthy_noise1_normz(:,:,l,n) = (C_healthy_noise1(:,:,l,n) - mu100(l,n))./std100(l,n);
            C_unhealthy_noise1_normz(:,:,l,n) = (C_unhealthy_noise1(:,:,l,n) - mu100(l,n))./std100(l,n);
            
    end
end



% KNN
% model 1
p = 0.9;      % proportion of rows to select for training
N = num_patients;  % total number of rows
tf_healthy1 = false(N,numel(k_decay),numel(k_gr),test_iter);    % create logical index vector
tf_healthy1(1:round(p*N),:,:,:) = true;

for l = 1:test_iter
    for n = 1:numel(k_gr)
        for m = 1:numel(k_decay)
            tf_healthy1(:,m,n) = tf_healthy1(randperm(N),m,n);   % randomise order
            C_healthy_noise1_train1(:,:,m,n) = C_healthy_noise1_norm1(:,squeeze(tf_healthy1(:,m,n)),m,n);
            C_healthy_noise1_test1(:,:,m,n) = C_healthy_noise1_norm1(:,squeeze(~tf_healthy1(:,m,n)),m,n);
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_train1(:,:,m,n) = C_unhealthy_noise1_norm1(:,squeeze(tf_healthy1(:,m,n)),m,n);
            C_unhealthy_noise1_test1(:,:,m,n) = C_unhealthy_noise1_norm1(:,squeeze(~tf_healthy1(:,m,n)),m,n);
            
            train_data1(:,:,m,n) = [C_healthy_noise1_train1(:,:,m,n),C_unhealthy_noise1_train1(:,:,m,n)];
            test_data1(:,:,m,n) = [C_healthy_noise1_test1(:,:,m,n),C_unhealthy_noise1_test1(:,:,m,n)];
            train_labels1(:,m,n) = [ones(1,p*N),zeros(1,p*N)];
            test_true_labels1(:,m,n) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
            
            [predicted_labels1(:,m,n),nn_index,accuracy1(:,m,n)] = KNN_(3,train_data1(:,:,m,n)',train_labels1(:,m,n)',test_data1(:,:,m,n)');
            [cm1(:,:,m,n,l),gn1(:,:,m,n,l)] = confusionmat(test_true_labels1(:,m,n)',predicted_labels1(:,m,n));
            % precision, recall and f1Scores
            precision1(:,m,n,l) = diag(cm1(:,:,m,n,l))./sum(cm1(:,:,m,n,l),2);
            recall1(:,m,n,l) = diag(cm1(:,:,m,n,l))./sum(cm1(:,:,m,n,l),1)';
            f1Scores1(:,m,n,l) = 2*(precision1(:,m,n,l).*recall1(:,m,n,l))./(precision1(:,m,n,l)+recall1(:,m,n,l));
            
        end
    end
end

meanF11 = mean(mean(f1Scores1,4),1);


figure;
h = surf(k_gr, k_decay, squeeze(meanF11));
title('5% noise - average subtraction');
xlabel('k_{growth} (day^{-1})'); ylabel('k_{decay} (day^{-1})');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
 set(h,'linestyle','none');
 
 %
tf_healthyz = false(N,numel(k_decay),numel(k_gr),test_iter);    % create logical index vector
tf_healthyz(1:round(p*N),:,:,:) = true;

for l = 1:test_iter
    for n = 1:numel(k_gr)
        for m = 1:numel(k_decay)
            tf_healthyz(:,m,n) = tf_healthyz(randperm(N),m,n);   % randomise order
            C_healthy_noise1_trainz(:,:,m,n) = C_healthy_noise1_normz(:,squeeze(tf_healthyz(:,m,n)),m,n);
            C_healthy_noise1_testz(:,:,m,n) = C_healthy_noise1_normz(:,squeeze(~tf_healthyz(:,m,n)),m,n);
            % for now using the same index of random selection for healthy and
            % unhealthy
            C_unhealthy_noise1_trainz(:,:,m,n) = C_unhealthy_noise1_normz(:,squeeze(tf_healthyz(:,m,n)),m,n);
            C_unhealthy_noise1_testz(:,:,m,n) = C_unhealthy_noise1_normz(:,squeeze(~tf_healthyz(:,m,n)),m,n);
            
            train_dataz(:,:,m,n) = [C_healthy_noise1_trainz(:,:,m,n),C_unhealthy_noise1_trainz(:,:,m,n)];
            test_dataz(:,:,m,n) = [C_healthy_noise1_testz(:,:,m,n),C_unhealthy_noise1_testz(:,:,m,n)];
            train_labelsz(:,m,n) = [ones(1,p*N),zeros(1,p*N)];
            test_true_labelsz(:,m,n) = [ones(1,round((1-p)*N)),zeros(1,round((1-p)*N))];
            
            [predicted_labelsz(:,m,n),nn_index,accuracyz(:,m,n)] = KNN_(3,train_dataz(:,:,m,n)',train_labelsz(:,m,n)',test_dataz(:,:,m,n)');
            [cmz(:,:,m,n,l),gnz(:,:,m,n,l)] = confusionmat(test_true_labelsz(:,m,n)',predicted_labelsz(:,m,n));
            % precision, recall and f1Scores
            precisionz(:,m,n,l) = diag(cmz(:,:,m,n,l))./sum(cmz(:,:,m,n,l),2);
            recallz(:,m,n,l) = diag(cmz(:,:,m,n,l))./sum(cmz(:,:,m,n,l),1)';
            f1Scoresz(:,m,n,l) = 2*(precisionz(:,m,n,l).*recallz(:,m,n,l))./(precisionz(:,m,n,l)+recallz(:,m,n,l));
            
        end
    end
end

meanF1z = mean(mean(f1Scoresz,4),1);


figure;
h = surf(k_gr, k_decay, squeeze(meanF1z));
title('5% noise - z-score normalization');
xlabel('k_{growth} (day^{-1})'); ylabel('k_{decay} (day^{-1})');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 3);
 set(h,'linestyle','none');