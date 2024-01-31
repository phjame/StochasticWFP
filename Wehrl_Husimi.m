%% Wigner-Fokker-Planck (SDE) equation using Euler-Maruyama method!
% initial state is: W0(q,p) = (2/h) * EXP[(-a^2*p^2/h_bar^2) - (q^2/a^2)]
% with h = 2*pi , a = 1.0 and h_bar = 1.0 (but in wigner for pedestrians
% h = 1) so the first term has SigmaP = 1/(sqrt(2)*a) and the second term
% has SigmaQ = a*sqrt(2) and both have mean = 0
clear;
clc;
%% Initializing
L = 1.0;
BinNum = 1000;
SigmaQ = L/sqrt(2); SigmaP = 1/(sqrt(2)*L);
s = 1/sqrt(2);
delta_t = 0.1;
Total_Time = 50;
NumOfTimeStep = round(Total_Time / delta_t);
NumOfParticles = 10^4;
D_qq = 1; D_pp = 1;
D = [D_qq , 0 ; 0 , D_pp];
mu1 = 0 ; mu2 = 0;
mu = [mu1 , mu2];
SSCovariance = [3., -1,;-1., 2.];
Entropy = zeros(NumOfTimeStep,1);
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
%% Wehrl entropy using Husimi function
State = zeros(1, NumOfTimeStep);
Time = zeros(1, NumOfTimeStep);
Time(1,1) = 0;
q1 = q(1,:)'; p1 = p(1,:)';
GMModel = fitgmdist([q1, p1],1);
GMModel.mu
GMModel.Sigma
A = GMModel.Sigma;
a = A(1,1); b = A(1,2); c = A(2,1); d = A(2,2);
State(1,1) = -1 + 5/(sqrt((-20 + b*(4 + b) - a*...
    (-4  + d) + 6*d)*(b^2 - a*d)));
AInv = A^(-1);
e = AInv(1,1); f = AInv(1,2); g = AInv(2,1); h = AInv(2,2);   
SigmaInv = [e*(h + 4 * s^2) - f^2, 4 * f * s^2; 4 * f * s^2, ...
    4 *((1 + e * s^2) * h - (f^2) * (s^2)) * s^2] / (h * (1 + e * s^2)...
    + (s^2) * (4 * (1 + e * s^2) - f^2));
S = SigmaInv^(-1);
Entropy(1) = log(det(S))/2.;

for k = 2:NumOfTimeStep
    qq = q(k,:)'; pp = p(k,:)';
    GMModel = fitgmdist([qq, pp],1);
    GMModel.mu
    GMModel.Sigma
    A = GMModel.Sigma;
    a = A(1,1); b = A(1,2); c = A(2,1); d = A(2,2);
    State(1,k) = -1 + 5/(sqrt((-20 + b*(4 + b) - a*...
        (-4  + d) + 6*d)*(b^2 - a*d)));
    AInv = A^(-1);
    e = AInv(1,1); f = AInv(1,2); g = AInv(2,1); h = AInv(2,2);   
    SigmaInv = [e*(h + 4 * s^2) - f^2, 4 * f * s^2; 4 * f * s^2, ...
        4 *((1 + e * s^2) * h - (f^2) * (s^2)) * s^2] / (h * (1 + e * s^2)...
        + (s^2) * (4 * (1 + e * s^2) - f^2));
    S = SigmaInv^(-1);
    Entropy(k) = log(det(S))/2.;
    Time(1,k) = Time(1,k-1) + delta_t;
end

tt = linspace(0,Total_Time,100);
Init_SState_norm = -1 + 5/(sqrt((-20 - (1/2)*(-4 + (1/2)) + ...
    6 * (1/2)) * (-(1/2) * (1/2))));
NormSigma = exp(-2*(1 - 1/sqrt(5))/2 * tt) * Init_SState_norm;

figure(7)
p = plot(Time , State,'o');
p.MarkerFaceColor = [1 1 1];
p.MarkerSize = 3;
p.MarkerEdgeColor = [1 0 0];

hold on
plot(tt,NormSigma,"blue", 'LineWidth', 2);
xlim([-0.2 50])
ylim([-0.1 1.6])
hold off

figure(8)
hold on
CovMatrix = SSCovariance;
CovMatrixInv = CovMatrix^(-1);
e = CovMatrixInv(1,1); f = CovMatrixInv(1,2); 
g = CovMatrixInv(2,1); h = CovMatrixInv(2,2);
SigmaInv = [e*(h + 4 * s^2) - f^2, 4 * f * s^2; 4 * f * s^2, ...
        4 *((1 + e * s^2) * h - (f^2) * (s^2)) * s^2] / (h * (1 + e * s^2)...
        + (s^2) * (4 * (1 + e * s^2) - f^2));
    
S = SigmaInv^(-1);
SSEntropy = log(det(S))/2.;
yline(SSEntropy)

plot(Time,Entropy);
hold off



