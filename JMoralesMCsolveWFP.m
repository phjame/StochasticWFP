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
sigmax=a/sqrt(2); sigmak=1/(a*sqrt(2));

Nsamples = 10^3; % <---- PARAMETER 1: VALUE DETERMINES NUMERICS CONVERGENCE

xold=zeros(1,Nsamples); kold=zeros(1,Nsamples); 
xnew=zeros(1,Nsamples); knew=zeros(1,Nsamples);
Nplot=1000;
%STEP 1: Sampling of Initial Condition, represented as point distribution.
for i=1:Nsamples
    xold(i)=normrnd(0,sigmax); kold(i)=normrnd(0,sigmak); %IC: Groundstate
end

% PLOT SAMPLING OF INITIAL CONDITION TOGETHER WITH ITS LEVEL SETS
figure(1); hold on; title('Initial Condition')
for i=1:Nsamples
    scatter(xold(i),kold(i),"red",'.')
end
% LEVEL SETS OF INITIAL CONDITION INCLUDED BELOW
f=@(x,k) (a^2)*(k.^2) + (x.^2)/(a^2);
dxplot = (max(xold)-min(xold))/Nplot;
dkplot = (max(kold)-min(kold))/Nplot;
xplot = min(xold):dxplot:max(xold);
kplot = min(kold):dkplot:max(kold);
[X,K]=meshgrid(xplot,kplot);
z=f(X,K);
contour(X,K,z,100)
hold off; 
exportgraphics(gcf,'InitialCondition.pdf','ContentType','vector')

%STEP 2: (k,-x-k)-Transport + D=Id-Diffusion over time evolution
Dxx = 1.; Dkk = 1.; D = [Dxx, 0.; 0., Dkk]; %Diffusion thru unit normals

T=10; dt=0.1; % <----- PARAMETERS 2 & 3: VALUES DETERMINE NUMERICAL CVG.

Ntime = round(T/dt);
% FWD EULER TIME EVOLUTION: 
%d(x,k)/dt = (k,-x-k) + randomwalk(D) so 
% (x,k)^new -(x,k)^old = dt*(k,-x-k) + dt*randomwalk(D) + O(dt^2) approx.
for j=1:Ntime
    for i=1:Nsamples
    	RandomVec=mvnrnd([0, 0],D*dt); %MC: Co-Variance matrix is dt*D
        xnew(i)= xold(i) + kold(i)*dt + RandomVec(1);  %normrnd(0,Dxx)*sqrt(dt);%D? E-Mry
        knew(i)= kold(i) -(xold(i)+kold(i))*dt + RandomVec(2); %normrnd(0,Dkk)*sqrt(dt);
    end
end

%PLOT SAMPLES OF NUMERICAL SOLUTION AND LEVEL SETS OF EXPECTED STEADY STATE
figure(2); hold on; title('Numerical Solution')
for i=1:Nsamples
    scatter(xnew(i),knew(i),"red",'.')
end
% LEVEL SETS OF STEADY STATE (INCLUDED IN THE FIGURE): GAUSSIAN ARGUMENT IS
fss=@(x,k) (3./10.)*(k.^2) + (x.^2)/5. + x.*k/5.;
dxplot = (max(xnew)-min(xnew))/Nplot;
dkplot = (max(knew)-min(knew))/Nplot;
xplot = min(xnew):dxplot:max(xnew);
kplot = min(knew):dkplot:max(knew);
[X,K]=meshgrid(xplot,kplot);
z=fss(X,K);
contour(X,K,z,100)
hold off;
exportgraphics(gcf,'NumericalSolution.pdf','ContentType','vector')

%ADDITIONAL STEP: Sampling of Steady State (SANITY CHECK).
% Analytical form of steady state solution is:
% exp(-x^2/5 - x.k/5 - 3k^2/10)/(2pi*sqrt(5))
% and for multivariate normal distributions we have the form
% exp(-(x-mu)Sigma^-1 (x-mu)'/2)/sqrt(|Sigma|(2pi)^d)
% with Sigma the covariance matrix.
% In this multivariate form, the steady state solution has a mu=0 and 
InvSigma = [2./5., 1./5.; 1./5., 3./5.]
%therefore it's the inverse of
Sigma = [3., -1,;-1., 2.]
Sigma*InvSigma
InvSigma*Sigma
for i=1:Nsamples
    RandomVec=mvnrnd([0, 0],Sigma); %Sampling from steady state
    xold(i)=RandomVec(1);
    kold(i)=RandomVec(2);
end
figure(3); hold on; title('Steady state')
for i=1:Nsamples
    scatter(xold(i),kold(i),"red",'.')
end
% LEVEL SETS OF STEADY STATE INCLUDED IN THE FIGURE
fss=@(x,k) (3./10.)*(k.^2) + (x.^2)/5. + x.*k/5.;
dxplot = (max(xold)-min(xold))/Nplot;
dkplot = (max(kold)-min(kold))/Nplot;
xplot = min(xold):dxplot:max(xold);
kplot = min(kold):dkplot:max(kold);
[X,K]=meshgrid(xplot,kplot);
z=fss(X,K);
contour(X,K,z,100);
hold off;
exportgraphics(gcf,'AnalyticalSteadyState.pdf','ContentType','vector')
