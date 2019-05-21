%% This code generates:
% a. noise-free curves for healthy and unhealthy cells with different
% kdecay and kgrowth rates.
% b. Make the plots for a range of kgrowth and kdecay
% c. Add three different type of noise to each curve
%  i. constant error
%  ii. standard fractional error
%  iii. fractional error
% d. KNN classification with simple noise
% e. 3 classes with different error
clear all; close all; clc;
%% a. noise-free growth curves
% defining parameters
time = (0:400)';
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(1,1) + Ch0;
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(1,1) + Ca0;
k_gr = 1e-2; 
k_decay = 1e-3;
C_healthy = zeros(1,numel(time));
C_unhealthy = zeros(1,numel(time));
for i = 1:numel(time)
    C_healthy(1,i) = Ch0_rnd;
    if i < 200
        C_unhealthy(1,i) = Ca0_rnd;
    else
        C_unhealthy(1,i) = Ca0_rnd*exp((k_gr/k_decay)*(1-exp(-k_decay*time(i-199))));
    end
end

figure;
plot(time, C_healthy, 'r', time, C_unhealthy, 'b--','LineWidth', 3);
legend('Healthy', 'Unhealthy'); legend boxoff
xlabel('time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',18,'FontName','Helvetica','YScale','log','LineWidth', 3);
%% b. Ranges for kgrowth and kdecay
clear all;close all; clc
% defining parameters for kgrowth
time = (0:400)';
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(10,1) + Ch0;
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(10,1) + Ca0;
k_gr = logspace(-4,0,10);
k_decay = 1e-3;
C_healthy_gr = zeros(numel(k_gr),numel(time));
C_unhealthy_gr = zeros(numel(k_gr),numel(time));

for j = 1:numel(k_gr)
    for i = 1:numel(time)
        C_healthy_gr(j,i) = Ch0_rnd(j);
        if i < 200
            C_unhealthy_gr(j,i) = Ca0_rnd(j);
        else
            C_unhealthy_gr(j,i) = Ca0_rnd(j)*exp((k_gr(j)/k_decay)*(1-exp(-k_decay*time(i-199))));
        end
    end
end

figure;
plot(time, C_healthy_gr, 'r', time, C_unhealthy_gr, 'b--','LineWidth', 3);
title('Range of K_{Growth}');
% legend(num2str(k_gr(1)),num2str(k_gr(2)),num2str(k_gr(3)),num2str(k_gr(4)),num2str(k_gr(5)),num2str(k_gr(6)),num2str(k_gr(7)),num2str(k_gr(8)),num2str(k_gr(9)),num2str(k_gr(10))); legend boxoff
xlabel('time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',18,'FontName','Helvetica','YScale','log','LineWidth', 3);

%%
clear all; close all; clc
% defining parameters for kgrowth
time = (0:400)';
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(10,1) + Ch0;
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(10,1) + Ca0;
k_gr = 1e-2;
k_decay = logspace(-4,2,10);
C_healthy_decay = zeros(numel(k_decay),numel(time));
C_unhealthy_decay = zeros(numel(k_decay),numel(time));
for j = 1:numel(k_decay)
    for i = 1:numel(time)
        C_healthy_decay(j,i) = Ch0_rnd(j);
        if i < 200
            C_unhealthy_decay(j,i) = Ca0_rnd(j);
        else
            C_unhealthy_decay(j,i) = Ca0_rnd(j)*exp((k_gr/k_decay(j))*(1-exp(-k_decay(j)*time(i-199))));
        end
    end
end

figure;
% plot(time, C_healthy_decay, 'r', time, C_unhealthy_decay, 'b--','LineWidth', 3);
plot( time, C_unhealthy_decay, 'b--','LineWidth', 3);
title('Range of K_{Decay}');
% legend('Healthy', 'Unhealthy'); legend boxoff
xlabel('time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',18,'FontName','Helvetica','YScale','log','LineWidth', 3);
%% c. noise added to the curves
%  i. constant error
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

C_healthy_noise1 = zeros(numel(time),num_patients);
C_unhealthy_noise1 = zeros(numel(time),num_patients);

for j = 1:num_patients
    for i = 1:numel(time)
        C_healthy(i,j) = Ch0_rnd(j);
        if i < 200
            C_unhealthy(i,j) = Ca0_rnd(j);
        else
            C_unhealthy(i,j) = Ca0_rnd(j)*exp((k_gr/k_decay)*(1-exp(-k_decay*time(i-199))));
        end
    end
end

for k = 1:num_patients
    
    C_healthy_noise1(:,k) = C_healthy(:,k)+5*rand(1,numel(C_healthy(:,k)))'.*C_healthy(:,k)/100;
    C_unhealthy_noise1(:,k) = C_unhealthy(:,k)+5*rand(1,numel(C_unhealthy(:,k)))'.*C_unhealthy(:,k)/100;
    
end
figure;
plot(time, C_healthy_noise1, 'r', time, C_unhealthy_noise1, 'b','LineWidth', 3);
% legend('Healthy', 'Unhealthy'); legend boxoff
xlabel('time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',18,'FontName','Helvetica','YScale','log','LineWidth', 3);

%  ii. standard fractional error


%  iii. fractional error
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
% model 2 (Matlab builtin code)
% outClass = knnclassify(sample, TRAIN, group, K, distance,rule)
[cm,gn] = confusionmat(test_true_labels',predicted_labels);
% precision, recall and f1Scores
precision = diag(cm)./sum(cm,2);

recall = diag(cm)./sum(cm,1)';

f1Scores = 2*(precision.*recall)./(precision+recall);

meanF1 = mean(f1Scores);
%% e. 3 classes with different error
% constant error model
clear all; close all; clc
time = (0:400)';
num_patients = 1;
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
k_gr = [1/60 1/18/30]; %day-1
k_decay_agg = linspace(1/150,1/30,10);
k_decay_nonagg = linspace(1/20/30,1/150,10);

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);

% C_healthy_noise1 = zeros(numel(time),num_patients);
% C_unhealthy_noise1 = zeros(numel(time),num_patients);
t_onset = 200;

C_healthy(:,1) = Ch0_rnd;


for k = 1:numel(k_decay_agg)
    for i = 1:numel(time)
        if i < t_onset
            C_unhealthy_agg(i,k) = Ch0_rnd;
        else
            C_unhealthy_agg(i,k) = Ch0_rnd*exp((k_gr(1)/k_decay_agg(k))*(1-exp(-k_decay_agg(k)*time(i-(t_onset-1)))));
        end
    end
end

for k = 1:numel(k_decay_nonagg)
    for i = 1:numel(time)
        if i < t_onset
            C_unhealthy_nonagg(i,k) = Ch0_rnd;
        else
            C_unhealthy_nonagg(i,k) = Ch0_rnd*exp((k_gr(2)/k_decay_nonagg(k))*(1-exp(-k_decay_nonagg(k)*time(i-(t_onset-1)))));
        end
    end
end

C_healthy_noise1 = C_healthy+15*rand(1,numel(C_healthy))'./100;

for l = 1:numel(k_decay_agg)
                
        C_unhealthy_agg_noise1(:,l) = C_unhealthy_agg(:,l)+15*rand(1,numel(C_unhealthy_agg(:,l)))'./100;
        C_unhealthy_nonagg_noise1(:,l) = C_unhealthy_nonagg(:,l)+15*rand(1,numel(C_unhealthy_nonagg(:,l)))'./100;

end

figure;
plot(time, C_healthy_noise1, 'r', time, C_unhealthy_agg_noise1, 'b',time, C_unhealthy_nonagg_noise1, 'g','LineWidth', 2);
title({'15% Noise - constant error model'});
legend('No Cancer', 'Aggressive', 'Non-Aggressive')
xlabel('Time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 4);
%%
% Standardize fractional model
clear all; close all; clc
time = (0:400)';
num_patients = 1;
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
k_gr = [1/60 1/18/30]; %day-1
k_decay_agg = linspace(1/150,1/30,10);
k_decay_nonagg = linspace(1/20/30,1/150,10);

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);

% C_healthy_noise1 = zeros(numel(time),num_patients);
% C_unhealthy_noise1 = zeros(numel(time),num_patients);
t_onset = 200;

C_healthy(:,1) = Ch0_rnd;


for k = 1:numel(k_decay_agg)
    for i = 1:numel(time)
        if i < t_onset
            C_unhealthy_agg(i,k) = Ch0_rnd;
        else
            C_unhealthy_agg(i,k) = Ch0_rnd*exp((k_gr(1)/k_decay_agg(k))*(1-exp(-k_decay_agg(k)*time(i-(t_onset-1)))));
        end
    end
end

for k = 1:numel(k_decay_nonagg)
    for i = 1:numel(time)
        if i < t_onset
            C_unhealthy_nonagg(i,k) = Ch0_rnd;
        else
            C_unhealthy_nonagg(i,k) = Ch0_rnd*exp((k_gr(2)/k_decay_nonagg(k))*(1-exp(-k_decay_nonagg(k)*time(i-(t_onset-1)))));
        end
    end
end

C_healthy_noise1 = C_healthy+15*rand(1,numel(C_healthy))'.*C_healthy/100;

for l = 1:numel(k_decay_agg)
                
        C_unhealthy_agg_noise1(:,l) = C_unhealthy_agg(:,l)+15*rand(1,numel(C_unhealthy_agg(:,l)))'.*C_unhealthy_agg(:,l)/100;
        C_unhealthy_nonagg_noise1(:,l) = C_unhealthy_nonagg(:,l)+15*rand(1,numel(C_unhealthy_nonagg(:,l)))'.*C_unhealthy_nonagg(:,l)/100;

end

figure;
plot(time, C_healthy_noise1, 'r', time, C_unhealthy_agg_noise1, 'b',time, C_unhealthy_nonagg_noise1, 'g','LineWidth', 2);
title({'5% Noise - Standardized Fractional error model'});
legend('No Cancer', 'Aggressive', 'Non-Aggressive')
xlabel('Time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 4);
%%
% fractional error model
clear all; close all; clc
time = (0:400)';
num_patients = 1;
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
k_gr = [1/60 1/18/30]; %day-1
k_decay_agg = linspace(1/150,1/30,10);
k_decay_nonagg = linspace(1/20/30,1/150,10);

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);

% C_healthy_noise1 = zeros(numel(time),num_patients);
% C_unhealthy_noise1 = zeros(numel(time),num_patients);
t_onset = 200;

C_healthy(:,1) = Ch0_rnd;


for k = 1:numel(k_decay_agg)
    for i = 1:numel(time)
        if i < t_onset
            C_unhealthy_agg(i,k) = Ch0_rnd;
        else
            C_unhealthy_agg(i,k) = Ch0_rnd*exp((k_gr(1)/k_decay_agg(k))*(1-exp(-k_decay_agg(k)*time(i-(t_onset-1)))));
        end
    end
end

for k = 1:numel(k_decay_nonagg)
    for i = 1:numel(time)
        if i < t_onset
            C_unhealthy_nonagg(i,k) = Ch0_rnd;
        else
            C_unhealthy_nonagg(i,k) = Ch0_rnd*exp((k_gr(2)/k_decay_nonagg(k))*(1-exp(-k_decay_nonagg(k)*time(i-(t_onset-1)))));
        end
    end
end

m = 50;
a = sqrt(0.5*m);
C_healthy_noise1 = C_healthy+(a*rand(1,numel(C_healthy))'+m).*C_healthy./100;

for l = 1:numel(k_decay_agg)
                
        C_unhealthy_agg_noise1(:,l) = C_unhealthy_agg(:,l)+(a*rand(1,numel(C_unhealthy_agg(:,l)))'+m).*C_unhealthy_agg(:,l)/100;
        C_unhealthy_nonagg_noise1(:,l) = C_unhealthy_nonagg(:,l)+(a*rand(1,numel(C_unhealthy_nonagg(:,l)))'+m).*C_unhealthy_nonagg(:,l)/100;

end

figure;
plot(time, C_healthy_noise1, 'r', time, C_unhealthy_agg_noise1, 'b',time, C_unhealthy_nonagg_noise1, 'g','LineWidth', 2);
title({'5% Noise - Fractional error model'});
legend('No Cancer', 'Aggressive', 'Non-Aggressive')
xlabel('Time (day)'); ylabel('Biomarker Conc. (ng/ml)');
set(gca,'box','off','TickDir','out',...
    'FontSize',24,'FontName','Helvetica','LineWidth', 4);