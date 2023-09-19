Nsamples = 10^4; % <---- PARAMETER 1: VALUE DETERMINES NUMERICS CONVERGENCE
Nplot=1000;
x=zeros(1,Nsamples); k=zeros(1,Nsamples); 
xmin=0.; xmax=0.;
kmin=0.; kmax=0.;
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
figure(3); hold on; title('Steady state')
for i=1:Nsamples
    RandomVec=mvnrnd([0, 0],Sigma); %Sampling from steady state
    xss=RandomVec(1);
    kss=RandomVec(2);
    scatter(xss,kss,5,"red",'.')
    if xss<xmin
	    xmin = xss;
    end
    if xss>xmax
	    xmax = xss;
    end
    if kss<kmin
	    kmin = kss;
    end
    if kss>kmax
	    kmax = kss;
    end
    x(1,i) = xss;
    k(1,i) = kss;
end
hist = histogram2(x,k,'DisplayStyle','tile','ShowEmptyBins','on');
% LEVEL SETS OF STEADY STATE INCLUDED IN THE FIGURE
fss=@(x,k) (3./10.)*(k.^2) + (x.^2)/5. + x.*k/5.;
dxplot = (xmax-xmin)/Nplot;
dkplot = (kmax-kmin)/Nplot;
xplot = xmin:dxplot:xmax;
kplot = kmin:dkplot:kmax;
[X,K]=meshgrid(xplot,kplot);
z=fss(X,K);
contour(X,K,z,100);%THIS PROVOKES THE ISSUE...?
hold off;
exportgraphics(gcf,'AnalyticalSteadyState.pdf','ContentType','vector')
