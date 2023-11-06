%% Wigner-Fokker-Planck (SDE) equation using Euler-Maruyama method!
% initial state is: W0(q,p) = (2/h) * EXP[(-a^2*p^2/h_bar^2) - (q^2/a^2)]
% with h = 2*pi , a = 1.0 and h_bar = 1.0 (but in wigner for pedestrians
% h = 1) so the first term has SigmaP = 1/(sqrt(2)*a) and the second term
% has SigmaQ = a*sqrt(2) and both have mean = 0
clear;
clc;
a = 1.0;
BinNum = 1000;
SigmaQ = a/sqrt(2); SigmaP = 1/(sqrt(2)*a);

%% Initializing
delta_t = 0.1;
Total_Time = 50;
NumOfTimeStep = round(Total_Time / delta_t);
NumOfParticles = 10^3;
D_qq = 1; D_pp = 1;
D = [D_qq , 0 ; 0 , D_pp];
mu1 = 0 ; mu2 = 0;
mu = [mu1 , mu2];

%% Initial State
q = zeros(NumOfTimeStep,NumOfParticles); 
p = zeros(NumOfTimeStep,NumOfParticles);

q(1,:) = normrnd(mu(1) , SigmaQ , [1, NumOfParticles]); 
p(1,:) = normrnd(mu(2) , SigmaP , [1, NumOfParticles]);

%% Transfer & Diffusion (Euler-Maruyama)
for i = 1 : NumOfTimeStep
    for j = 1 : NumOfParticles
        epsilon = mvnrnd(mu, 2*D*delta_t);
        q(i+1,j) = q(i,j) + p(i,j)*delta_t + epsilon(1);
        p(i+1,j) = p(i,j) + (-q(i,j) - p(i,j))*delta_t + epsilon(2);
    end
end

%%
State = zeros(1, round(numel(q(:,1))/10));
l = 1;
Time = zeros(1, length(1:10:numel(q(:,1))));

State(1,l) = -1 + 5/(sqrt((-20 - (1/sqrt(2))*(-4 + (1/sqrt(2))) + ...
    6 * (1/sqrt(2))) * (-(1/sqrt(2)) * (1/sqrt(2)))));

for k = 11:10:numel(q(:,1))

    l = l + 1;
    qq = q(k,:)'; pp = p(k,:)';
    GMModel = fitgmdist([qq, pp],1);
    GMModel.mu
    GMModel.Sigma
    A = GMModel.Sigma;
    State(1,l) = -1 + 5/(sqrt((-20 + A(1,2)*(4 + A(1,2)) - A(1,1)*...
        (-4  + A(2,2)) + 6*A(2,2))*(A(1,2)^2 - A(1,1)*A(2,2))));

    
    t = k * delta_t;
    Time(1,l) = t;
    
end

% plotting the upper bound! Initial state - SS
tt = linspace(0,Total_Time,100);
Init_SState = -1 + 5/(sqrt((-20 - (1/sqrt(2))*(-4 + (1/sqrt(2))) + ...
    6 * (1/sqrt(2))) * (-(1/sqrt(2)) * (1/sqrt(2)))));

% plotting the solutions
figure(6)
YSigma1 = exp(-0.2*(1 - 1/sqrt(5))/2 * tt) * Init_SState;
plot(tt,YSigma1)
hold on
YSigma2 = exp(-0.5*(1 - 1/sqrt(5))/2 * tt) * Init_SState;
plot(tt,YSigma2)
YSigma3 = exp(-(1 - 1/sqrt(5))/2 * tt) * Init_SState;
plot(tt,YSigma3)
plot(Time , State,'o')
hold off
%% Fit Exponential Function

f = fit(Time',State','exp1')
figure(7)
plot(f,Time,State)

