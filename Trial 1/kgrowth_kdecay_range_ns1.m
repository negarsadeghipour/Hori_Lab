%% k_gr_non
clear all; close all; clc
time = (0:400)';
num_patients = 1;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
s_mean_k_gr_non = linspace(0,0.01,5);
max_k_gr_non = log(2)/18/30 + s_mean_k_gr_non;
min_k_gr_non = 0 + s_mean_k_gr_non;
max_k_decay_non = log(2)/5/30;
min_k_decay_non = log(2)/24/30;
k_gr_non_rnd = (max_k_gr_non - min_k_gr_non).*rand(num_patients,1) + min_k_gr_non;%mean([0 1/18/30]);
k_decay_non_rnd = (max_k_decay_non - min_k_decay_non).*rand(num_patients,1) + min_k_decay_non;%mean([1/(24*30) 1/150]);


%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
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

for j = 1:numel(s_mean_k_gr_non)
    for i = 1:numel(time)
        if i < t_onset
            C_healthy(i,j) = Ch0_rnd;
            C_unhealthy(i,j) = Ca0_rnd;
        else
            C_healthy(i,j) = Ch0_rnd*exp((k_gr_non_rnd(j)/k_decay_non_rnd)*(1-exp(-k_decay_non_rnd*time(i-(t_onset-1)))));
            C_unhealthy(i,j) = Ca0_rnd*exp((k_gr_agg_rnd/k_decay_agg_rnd)*(1-exp(-k_decay_agg_rnd*time(i-(t_onset-1)))));
        end
    end
end

C_healthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));
C_unhealthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));

for n = 1:numel(s_mean_k_gr_non)
        
        C_healthy_noise1(:,n) = C_healthy(:,n)+noise_i*rand(1,numel(C_healthy(:,n)))'.*C_healthy(:,n)/100;
        C_unhealthy_noise1(:,n) = C_unhealthy(:,n)+noise_i*rand(1,numel(C_unhealthy(:,n)))'.*C_unhealthy(:,n)/100;
        
end
figure;plot(time,squeeze(C_healthy_noise1(:,:)),'r-','LineWidth', 2);%,time,squeeze(C_unhealthy_noise1(:,1,:)),'b-'); 
title('Non-aggressive - k_{Growth}');
xlabel('Time (day)'); ylabel('Biomarker level (ng/mL)');
set(gca,'box','off','TickDir','out',...
    'FontSize',20,'FontName','Helvetica','LineWidth', 3);
%% k_decay_non
clear all; close all; clc
time = (0:400)';
num_patients = 1;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
max_k_gr_non = log(2)/18/30;
min_k_gr_non = 0;

s_mean_k_decay_non = linspace(0,0.1,5);
max_k_decay_non = log(2)/5/30 + s_mean_k_decay_non;
min_k_decay_non = log(2)/24/30 + s_mean_k_decay_non;
k_gr_non_rnd = (max_k_gr_non - min_k_gr_non).*rand(num_patients,1) + min_k_gr_non;%mean([0 1/18/30]);
k_decay_non_rnd = (max_k_decay_non - min_k_decay_non).*rand(num_patients,1) + min_k_decay_non;%mean([1/(24*30) 1/150]);


%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
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

for j = 1:numel(s_mean_k_decay_non)
    for i = 1:numel(time)
        if i < t_onset
            C_healthy(i,j) = Ch0_rnd;
            C_unhealthy(i,j) = Ca0_rnd;
        else
            C_healthy(i,j) = Ch0_rnd*exp((k_gr_non_rnd/k_decay_non_rnd(j))*(1-exp(-k_decay_non_rnd(j)*time(i-(t_onset-1)))));
            C_unhealthy(i,j) = Ca0_rnd*exp((k_gr_agg_rnd/k_decay_agg_rnd)*(1-exp(-k_decay_agg_rnd*time(i-(t_onset-1)))));
        end
    end
end

C_healthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));
C_unhealthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));

for n = 1:numel(s_mean_k_decay_non)
        
        C_healthy_noise1(:,n) = C_healthy(:,n)+noise_i*rand(1,numel(C_healthy(:,n)))'.*C_healthy(:,n)/100;
        C_unhealthy_noise1(:,n) = C_unhealthy(:,n)+noise_i*rand(1,numel(C_unhealthy(:,n)))'.*C_unhealthy(:,n)/100;
        
end
figure;plot(time,squeeze(C_healthy_noise1(:,:)),'g-','LineWidth', 2);%,time,squeeze(C_unhealthy_noise1(:,1,:)),'b-'); 
title('Non-aggressive - k_{Decay}');
xlabel('Time (day)'); ylabel('Biomarker level (ng/mL)');
set(gca,'box','off','TickDir','out',...
    'FontSize',20,'FontName','Helvetica','LineWidth', 3);
%% k_gr
clear all; close all; clc
time = (0:400)';
num_patients = 1;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;

max_k_gr_non = log(2)/18/30;
min_k_gr_non = 0;
max_k_decay_non = log(2)/5/30;
min_k_decay_non = log(2)/24/30;
k_gr_non_rnd = (max_k_gr_non - min_k_gr_non).*rand(num_patients,1) + min_k_gr_non;%mean([0 1/18/30]);
k_decay_non_rnd = (max_k_decay_non - min_k_decay_non).*rand(num_patients,1) + min_k_decay_non;%mean([1/(24*30) 1/150]);

%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
s_mean_k_gr_agg = linspace(0,0.02,5);
max_k_gr_agg = log(2)/2/30 + s_mean_k_gr_agg;
min_k_gr_agg = log(2)/18/300 + s_mean_k_gr_agg;
max_k_decay_agg = log(2)/30;
min_k_decay_agg = log(2)/5/30;
k_gr_agg_rnd = (max_k_gr_agg - min_k_gr_agg).*rand(num_patients,1) + min_k_gr_agg;%linspace(1/18/30,1/2/30,50); %day-1
k_decay_agg_rnd = (max_k_decay_agg - min_k_decay_agg).*rand(num_patients,1) + min_k_decay_agg;%linspace(1/150,1/30,50); %day-1

m_k_gr_agg_rnd = mean(k_gr_agg_rnd);
sd_k_gr_agg_rnd = std(k_gr_agg_rnd);

m_k_decay_agg_rnd = mean(k_decay_agg_rnd);
sd_k_decay_agg_rnd = std(k_decay_agg_rnd);

noise_i = 0;

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);
t_onset = 200;

for j = 1:numel(s_mean_k_gr_agg)
    for i = 1:numel(time)
        if i < t_onset
            C_healthy(i,j) = Ch0_rnd;
            C_unhealthy(i,j) = Ca0_rnd;
        else
            C_healthy(i,j) = Ch0_rnd*exp((k_gr_non_rnd/k_decay_non_rnd)*(1-exp(-k_decay_non_rnd*time(i-(t_onset-1)))));
            C_unhealthy(i,j) = Ca0_rnd*exp((k_gr_agg_rnd(j)/k_decay_agg_rnd)*(1-exp(-k_decay_agg_rnd*time(i-(t_onset-1)))));
        end
    end
end

C_healthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));
C_unhealthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));

for n = 1:numel(s_mean_k_gr_agg)
        
        C_healthy_noise1(:,n) = C_healthy(:,n)+noise_i*rand(1,numel(C_healthy(:,n)))'.*C_healthy(:,n)/100;
        C_unhealthy_noise1(:,n) = C_unhealthy(:,n)+noise_i*rand(1,numel(C_unhealthy(:,n)))'.*C_unhealthy(:,n)/100;
        
end
figure;plot(time,squeeze(C_unhealthy_noise1(:,:)),'b-','LineWidth', 2);%,time,squeeze(C_unhealthy_noise1(:,1,:)),'b-'); 
title('Aggressive - k_{Growth}');
xlabel('Time (day)'); ylabel('Biomarker level (ng/mL)');
set(gca,'box','off','TickDir','out',...
    'FontSize',20,'FontName','Helvetica','LineWidth', 3);
%% k_decay_non
clear all; close all; clc
time = (0:400)';
num_patients = 1;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
max_k_gr_non = log(2)/18/30 ;
min_k_gr_non = 0;
max_k_decay_non = log(2)/5/30;
min_k_decay_non = log(2)/24/30;
k_gr_non_rnd = (max_k_gr_non - min_k_gr_non).*rand(num_patients,1) + min_k_gr_non;%mean([0 1/18/30]);
k_decay_non_rnd = (max_k_decay_non - min_k_decay_non).*rand(num_patients,1) + min_k_decay_non;%mean([1/(24*30) 1/150]);


%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
Ca0_rnd = std_a.*randn(num_patients,1) + Ca0;
max_k_gr_agg = log(2)/2/30;
min_k_gr_agg = log(2)/18/300;
s_mean_k_decay_agg = linspace(0,0.1,5);
max_k_decay_agg = log(2)/30 + s_mean_k_decay_agg;
min_k_decay_agg = log(2)/5/30 + s_mean_k_decay_agg;
k_gr_agg_rnd = (max_k_gr_agg - min_k_gr_agg).*rand(num_patients,1) + min_k_gr_agg;%linspace(1/18/30,1/2/30,50); %day-1
k_decay_agg_rnd = (max_k_decay_agg - min_k_decay_agg).*rand(num_patients,1) + min_k_decay_agg;%linspace(1/150,1/30,50); %day-1


noise_i = 0;

C_healthy = zeros(numel(time),num_patients);
C_unhealthy = zeros(numel(time),num_patients);
t_onset = 200;

for j = 1:numel(s_mean_k_decay_agg)
    for i = 1:numel(time)
        if i < t_onset
            C_healthy(i,j) = Ch0_rnd;
            C_unhealthy(i,j) = Ca0_rnd;
        else
            C_healthy(i,j) = Ch0_rnd*exp((k_gr_non_rnd/k_decay_non_rnd)*(1-exp(-k_decay_non_rnd*time(i-(t_onset-1)))));
            C_unhealthy(i,j) = Ca0_rnd*exp((k_gr_agg_rnd/k_decay_agg_rnd(j))*(1-exp(-k_decay_agg_rnd(j)*time(i-(t_onset-1)))));
        end
    end
end

C_healthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));
C_unhealthy_noise1 = zeros(numel(time),num_patients,numel(noise_i));

for n = 1:numel(s_mean_k_decay_agg)
        
        C_healthy_noise1(:,n) = C_healthy(:,n)+noise_i*rand(1,numel(C_healthy(:,n)))'.*C_healthy(:,n)/100;
        C_unhealthy_noise1(:,n) = C_unhealthy(:,n)+noise_i*rand(1,numel(C_unhealthy(:,n)))'.*C_unhealthy(:,n)/100;
        
end
figure;plot(time,squeeze(C_unhealthy_noise1(:,:)),'k-','LineWidth', 2);%,time,squeeze(C_unhealthy_noise1(:,1,:)),'b-'); 
title('Aggressive - k_{Decay}');
xlabel('Time (day)'); ylabel('Biomarker level (ng/mL)');
set(gca,'box','off','TickDir','out',...
    'FontSize',20,'FontName','Helvetica','LineWidth', 3);
%% example of four within accepted range
clear all; close all; clc
time = (0:400)';
num_patients = 10;
%healthy baseline and std
Ch0 = 8; %ng/mL
std_h = 1.5; %ng/mL
Ch0_rnd = std_h.*randn(num_patients,1) + Ch0;
max_k_gr_non = log(2)/18/30;
min_k_gr_non = 0 + 0.000001;
max_k_decay_non = log(2)/5/30;
min_k_decay_non = log(2)/24/30;
k_gr_non_rnd = (max_k_gr_non - min_k_gr_non).*rand(num_patients,1) + min_k_gr_non;%mean([0 1/18/30]);
k_decay_non_rnd = (max_k_decay_non - min_k_decay_non).*rand(num_patients,1) + min_k_decay_non;%mean([1/(24*30) 1/150]);


%unhealthy baseline and std
Ca0 = 8; %ng/mL
std_a = 1.5; %ng/mL
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
            C_healthy(i,j) = Ch0_rnd(j)*exp((k_gr_non_rnd(j)/k_decay_non_rnd(j))*(1-exp(-k_decay_non_rnd(j)*time(i-(t_onset-1)))));
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

figure;plot(time,squeeze(C_healthy_noise1(:,:)),'r-',time,squeeze(C_unhealthy_noise1(:,:)),'b-','LineWidth', 2);%,time,squeeze(C_unhealthy_noise1(:,1,:)),'b-'); 
title('');
xlabel('Time (day)'); ylabel('Biomarker level (ng/mL)');
set(gca,'box','off','TickDir','out',...
    'FontSize',20,'FontName','Helvetica','LineWidth', 3);