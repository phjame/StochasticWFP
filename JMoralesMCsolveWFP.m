%Monte Carlo solver for Wigner-Fokker-Planck equation (units st m=1,hbar=1)

% in Gradient form: 
% w_t + k.w_x -x.w_k = w_xx + w_kk + div_k.(kw)
% in Divergence form: 
% w_t + div_(x,k) .{w(k,-x-k)} = w_xx + w_kk

% Transport vector: (dx/dt,dk/dt)=(k,-x-k)

% Diffusion Matrix: D = Id, the identity matrix (unit normals in x,k)

% INITIAL CONDITION: A Coherent State. Specifically,  Harmonic Groundstate
% W0(x,p) = (2/h) exp(-a^2 k^2 - x^2/a^2) = (2/h)exp(-x^2/a^2)exp(-a^2 k^2)

% Since normals are f(x)=exp(-0.5[(x-mu)/sigma]^2)/(sqrt(2pi)sigma), then
% the quasi-distribution given by the groundstate and proportional to 
% exp(-x^2/a^2) has sigma=a/sqrt(2), exp(-a^2 k^2) has sigma=1/(sqrt(2)a)

L=1.0; %h=2*pi; %1=hbar=h/(2pi) => h=2*pi
%sigmax=L/2.; sigmak=1./L; %Squeezed state parameters
%sigmax=L/sqrt(2); sigmak=1/(L*sqrt(2)); %Harmonic groundstate parameters
%sigmax=2.25.*L/sqrt(2); sigmak=2.25.*1./(L*sqrt(2));%diffus/varianz/uncrtn

Nsamples = 10^4; % <---- PARAMETER 1: VALUE DETERMINES NUMERICS CONVERGENCE
T=50; dt=0.01; % <----- PARAMETERS 2 & 3: VALUES DETERMINE NUMERICAL CVG.
Ntime = round(T/dt);
x=zeros(Ntime,Nsamples); k=zeros(Ntime,Nsamples); 
Entropy=zeros(Ntime,1); weightL2norm = zeros(Ntime,1); antisym = zeros(Ntime,1)
Nplot=1000; Sigma = [3., -1.;-1., 2.]
%STEP 1: Sampling of Initial Condition, represented as point distribution.
for i=1:Nsamples
%    x(1,i)=normrnd(0,sigmax); k(1,i)=normrnd(0,sigmak); %IC: Groundstate
    RandomVec=mvnrnd([0, 0],1.5*Sigma); %Sampling from coeff*steady-state
    x(1,i)=RandomVec(1);
    k(1,i)=RandomVec(2);
end
s = 1/sqrt(2);
% PLOT SAMPLING OF INITIAL CONDITION TOGETHER WITH ITS LEVEL SETS
figure(1); hold on; title('Initial Condition')
for i=1:Nsamples
    scatter(x(1,i),k(1,i),5,"red",'.')
end
hist = histogram2(x(1,:),k(1,:),'DisplayStyle','tile','ShowEmptyBins','on')
GMModel = fitgmdist([x(1,:)',k(1,:)'],1)
GMModel.mu
GMModel.Sigma
CovMatrixIC = GMModel.Sigma
antisym(1) = CovMatrixIC(1,2)-CovMatrixIC(2,1)
%CovMatrixIC = (CovMatrixIC + CovMatrixIC')*0.5;
CovMatrixInv = CovMatrixIC^(-1)
a=CovMatrixInv(1,1); b=CovMatrixInv(1,2); c=CovMatrixInv(2,1); d=CovMatrixInv(2,2);
Sinv = [a*(d+4*s^2)-b^2, 4*b*s^2; 4*b*s^2, 4*((1+a*s^2)*d-(b^2)*(s^2))*s^2]
Sinv = Sinv/(d*(1+a*s^2)+(s^2)*(4*(1+a*s^2)-b^2));
S = Sinv^(-1)
Entropy(1) = log(det(S))/2.;
weightL2norm(1) = sqrt(-1+(sqrt(5)/(sqrt(det(2*CovMatrixIC^(-1)-Sigma^(-1)))*det(CovMatrixIC))))
gmPDF = @(x,k) arrayfun(@(x0,k0) pdf(GMModel,[x0 k0]),x(1,:)',k(1,:)')
gfun = gca
fcontour(gmPDF,[gfun.XLim gfun.YLim],'--r')
% LEVEL SETS OF INITIAL CONDITION INCLUDED BELOW
f=@(x,k) (L^2)*(k.^2) + (x.^2)/(L^2);
dxplot = (max(x(1,:))-min(x(1,:)))/Nplot;
dkplot = (max(k(1,:))-min(k(1,:)))/Nplot;
xplot = min(x(1,:)):dxplot:max(x(1,:));
kplot = min(k(1,:)):dkplot:max(k(1,:));
[X,K]=meshgrid(xplot,kplot);
z=f(X,K);
contour(X,K,z,100)
hold off; 
%exportgraphics(gcf,'InitialCondition.pdf','ContentType','image')

%STEP 2: (k,-x-k)-Transport + D=Id-Diffusion over time evolution
Dxx = 1.; Dkk = 1.; D = [Dxx, 0.; 0., Dkk]; %Diffusion thru unit normals

% FWD EULER TIME EVOLUTION: 
%d(x,k)/dt = (k,-x-k) + randomwalk(D) so 
% (x,k)^new -(x,k)^old = dt*(k,-x-k) + dt*randomwalk(D) + O(dt^2) approx.
for j=2:Ntime
    j
    for i=1:Nsamples
    	RandomVec=mvnrnd([0, 0],2*D*dt); %MC: Co-Variance matrix is dt*D
        x(j,i)= x(j-1,i) + k(j-1,i)*dt + RandomVec(1);  %normrnd(0,Dxx)*sqrt(dt);%D? E-Mry
        k(j,i)= k(j-1,i) -(x(j-1,i)+k(j-1,i))*dt + RandomVec(2); %normrnd(0,Dkk)*sqrt(dt);
%        figure(3); title('Time evolution') 
%        scatter(x(j,i),k(j,i),5,"red",'.')
    end
    GMModel = fitgmdist([x(j,:)',k(j,:)'],1) 
    GMModel.mu 
    GMModel.Sigma
    CovMatrix = GMModel.Sigma
    antisym(j) = CovMatrix(1,2)-CovMatrix(2,1)
%    CovMatrix = 0.5*(CovMatrix+CovMatrix')
    CovMatrixInv = CovMatrix^(-1)
    a=CovMatrixInv(1,1); b=CovMatrixInv(1,2); c=CovMatrixInv(2,1); d=CovMatrixInv(2,2);
    Sinv = [a*(d+4*s^2)-b^2, 4*b*s^2; 4*b*s^2, 4*((1+a*s^2)*d-(b^2)*(s^2))*s^2]
    Sinv = Sinv/(d*(1+a*s^2)+(s^2)*(4*(1+a*s^2)-b^2));
    S = Sinv^(-1)
    Entropy(j) = log(det(S))/2.;
    weightL2norm(j) = sqrt(-1+(sqrt(5)/(sqrt(det(2*CovMatrix^(-1)-Sigma^(-1)))*det(CovMatrix))));
end
NormAsym = norm(antisym)
Entropy(Ntime)
CovMatrix = Sigma
CovMatrixInv = CovMatrix^(-1)
a=CovMatrixInv(1,1); b=CovMatrixInv(1,2); c=CovMatrixInv(2,1); d=CovMatrixInv(2,2);
Sinv = [a*(d+4*s^2)-b^2, 4*b*s^2; 4*b*s^2, 4*((1+a*s^2)*d-(b^2)*(s^2))*s^2]
Sinv = Sinv/(d*(1+a*s^2)+(s^2)*(4*(1+a*s^2)-b^2));
S = Sinv^(-1)
EntropySS = log(det(S))/2.;
figure(4); hold on; 
%plot((1:1:Ntime)*dt,Entropy,'.')
errorbar((1:1:Ntime)*dt,Entropy,0.03*ones(size(Entropy)),"-s",...
    "MarkerEdgeColor","blue","MarkerFaceColor",[0.65 0.85 0.90])
xlabel('Time (t)')
ylabel('Entropy (H)')
yline(EntropySS)
title('Entropy (H) vs Time (t)')
legend({'Entropy of Numerical Solution','Analytical Steady State Entropy'},'Location','best')
hold off;
figure(5); hold on;
plot((1:1:Ntime)*dt,weightL2norm,'.')
plot((1:1:Ntime)*dt,weightL2norm(1)*exp(-((1:1:Ntime)*dt)*(1-1/sqrt(5))/2),'r')
xlabel('Time (t)')
ylabel('L2mu')
legend({'Weighted L2 distance to Steady State','Exponential Decay Rate Bound'},'Location','best')
title('Weighted L2 distance to Steady state (L2mu)')
hold off;
%PLOT SAMPLES OF NUMERICAL SOLUTION AND LEVEL SETS OF EXPECTED STEADY STATE
figure(2); hold on; title('Numerical Solution')
for i=1:Nsamples
    scatter(x(Ntime,i),k(Ntime,i),5,"red",'.');
end
hist = histogram2(x(Ntime,:),k(Ntime,:),'DisplayStyle','tile','ShowEmptyBins','on')
GMModel = fitgmdist([x(Ntime,:)',k(Ntime,:)'],1)
GMModel.mu
GMModel.Sigma
CovMatrixSS = GMModel.Sigma
gmPDF = @(x,k) arrayfun(@(x0,k0) pdf(GMModel,[x0 k0]),x(Ntime,:)',k(Ntime,:)')
gfun = gca
fcontour(gmPDF,[gfun.XLim gfun.YLim],'--r')
% LEVEL SETS OF STEADY STATE (INCLUDED IN THE FIGURE): GAUSSIAN ARGUMENT IS
fss=@(x,k) (3./10.)*(k.^2) + (x.^2)/5. + x.*k/5.;
dxplot = (max(x(Ntime,:))-min(x(Ntime,:)))/Nplot;
dkplot = (max(k(Ntime,:))-min(k(Ntime,:)))/Nplot;
xplot = min(x(Ntime,:)):dxplot:max(x(Ntime,:));
kplot = min(k(Ntime,:)):dkplot:max(k(Ntime,:));
[X,K]=meshgrid(xplot,kplot);
z=fss(X,K);
contour(X,K,z,100)
hold off;
exportgraphics(gcf,'NumericalSolution.pdf','ContentType','image')
