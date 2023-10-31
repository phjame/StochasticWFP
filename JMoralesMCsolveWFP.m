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

a=1.0; %h=2*pi; %1=hbar=h/(2pi) => h=2*pi
sigmax=a/sqrt(2); sigmak=1/(a*sqrt(2)); %Harmonic oscillator parameters
%sigmax=0.01*a/sqrt(2); sigmak=0.01/(a*sqrt(2));%Small diffus/varianz/uncrtn

Nsamples = 10^4; % <---- PARAMETER 1: VALUE DETERMINES NUMERICS CONVERGENCE
T=50; dt=0.01; % <----- PARAMETERS 2 & 3: VALUES DETERMINE NUMERICAL CVG.
Ntime = round(T/dt);

x=zeros(Ntime,Nsamples); k=zeros(Ntime,Nsamples); Entropy=zeros(Ntime,1);
Nplot=1000; Sigma = [3., -1,;-1., 2.]
%STEP 1: Sampling of Initial Condition, represented as point distribution.
for i=1:Nsamples
    x(1,i)=normrnd(0,sigmax); k(1,i)=normrnd(0,sigmak); %IC: Groundstate
%    RandomVec=mvnrnd([0, 0],0.01*Sigma); %Sampling from coeff*steady-state
%    x(1,i)=RandomVec(1);
%    k(1,i)=RandomVec(2);
end

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
Entropy(1) = log(det(CovMatrixIC))/2.;
gmPDF = @(x,k) arrayfun(@(x0,k0) pdf(GMModel,[x0 k0]),x(1,:)',k(1,:)')
gfun = gca
fcontour(gmPDF,[gfun.XLim gfun.YLim],'--r')
% LEVEL SETS OF INITIAL CONDITION INCLUDED BELOW
f=@(x,k) (a^2)*(k.^2) + (x.^2)/(a^2);
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
    for i=1:Nsamples
	j
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
    Entropy(j) = log(det(CovMatrix))/2.;
end
Entropy(Ntime)
EntropySS = log(5)/2.
figure(4); hold on;
plot((1:1:Ntime)*dt,Entropy,'.')
xlabel('Time (t)')
ylabel('Entropy (H)')
yline(EntropySS)
title('Entropy (H) vs Time (t)')
legend({'Entropy of Numerical Solution','Analytical Steady State Entropy'},'Location','best')
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
