% mixed_logit_estimator1: estimation/identification of underling distribution of random coefficients
%                         of a simple example binary logit model under alternative distributions for
%                         unobserved heterogeneity
% 
%                         Similar to mixed_logit_estimator.m except that intercept is treated as a random coefficient
%                         in addition to the slope of the x covariate (think of x as price)
%
%                         John Rust, July 2024


global nqp qw qa data;

data=load('choice_data.dat');  % load the data set of simulated choices, a 1000 x 4 matrix where
                               % first column is the covariate, x, and the final 3 columns are 0/1
                               % choice dummies (dependent variables) for 3 different specifications
                               % for the distribution of the random coefficient beta of x

% generate quadrature weights, abscissae for normally distributed random coefficient on x
% in the binary logit model.


nqp=80;
[qa,qw]=quadpoints(nqp,0,1);
qa=icdf('normal',qa,0,1);

% the code below is for an analysis of the identification of the model: it calculates the
% nonlinear least squares estimates of the parameters of different specifications for the
% distribution of random coefficients F(beta) for different domains of the x covariate and
% allows for the estimated F(beta) to be different from the true one to see how closely
% we can approximate a true F(beta) using a misspecified F(beta) distribution, such as
% to judge how well a finite mixture specification can approximate a continuous Normal 
% distribution for F(beta) and vice versa. For discrete mixture specifications there can
% be up to 5 points of support.

identification_analysis=1;  % set this to 1 to use nonlinear least squares to find a best-fitting 
                            % distribution F(beta) to the true specification, using the true P(x) (probability of 1 given x)
                            % (unconditional choice probability of alternative d=1, after integrating
                            % out the random coefficient beta) as the dependent variable in the regression

ml_estimation=0;            % set this to 1 to use maximum likelihood to estimate your chosen model (either 
                            % continuous normally distributed random coefficient F(beta), or a finite mixture
                            % specification by maximum likelihood

gendat=0;                   % enter 1 to generate new x covariates using the nobs (number of observations) variable below.
                            % Otherwise the program will use the x vector in memory, if it exists or load a copy from disk otherwise

nobs=5000;                % number of observations for the estimation. 

nreplications=1;            % number of replications for a Monte carlo analysis of the estimated

regen_true_model=0;         % Enter 1 to randomly draw parameters for the "true model" of unobserved types/random coefficients,
                            % otherwise the previously drawn values (stored in thetatrue vector) will be used if present in memory.

newstart=0;                 % Enter 1 to start the optimization from a randomly initialized parameter value. 
                            % If newstart=0 then whatever value of thetastart is in memory (a value from a previous run) is used

objective='kldistance';     % when identification_analysis=1, this specifies the objective to be minimized to uncover the distribution
                            % of random coefficients using a multinomial distribution with k "types" of individuals (i.e. a single layer neural network
                            % with a logistic or softmax transfer function and k hidden units). 
                            % set objective='kldistance' to minimize the Kullback-Leibler distance between the true CCP (with continuously
                            %       distributed coefficients, potentially) and the approximate CCP using the finite mixture of types
                            % set objective='leastsquares' to minimize the squared error prediction error between the true CCP and approximate one

truemixing='continuous';    % a flag to specify the true mixing distribution used to generate P(x): continuous vs discrete
estmixing='discrete';       % a flag to specify what type of model to estimate for F(beta): continuyous vs discrete

T=50;                       % length of panel. Set T=1 for pure cross sectional data.

probtype='bprob1';          % set the function to call to calculate the mixed choice probabilities entering the mixed CCP. Only
                            % relevant in the cross section case, T=1, since when T > 1, the function bprob2 is always used to
                            % calculate the probabilites in the panel likelihood observation by observation

n_est_types=1;              % if estmixing='discrete' this is the number of discrete mass points in the finite mixture approximation to F(beta)
                            % if estmixing='continuous' and n_est_types > 1, then a mixture of n_est_types normals is estimated
n_true_types=2;             % if truemixing='discrete' then this gives the number of discrete mass points for F(beta) used to
                            % calculate true choice probability P(x). If truemixing or estmixing is continuous then if n_true_types=1
                            % 3 parameters are estimated: theta(1)=alpha (the constant term in the logit) and mu=theta(2) and
                            % sigma=exp(theta(3)), there (mu,sigma^2) are the mean and variance of a continuous normally
                            % distributed specification for F(beta).
                            %
                            % If n_true_types > 1 then a mixture of normals is used for the mixing distribution F(beta) where
                            % beta is a 2x1 bivariate normal random variable whose first element beta(1) is the random intercept and
                            % beta(2) is the random slope coefficient for the covariate x. There are 5 parameters necessary to specify
                            % the mean and covariance matrix of a bivariate normal distribution so the first 5*n_true_types parameters are these
                            % vectors stacked by each of the n_true_types. The remaining n_true_types-1 parameters are parameters for a multinomial
                            % logit model for the mixture weights, so beta will be a mixture of bivariate normals with n_true_types mixture components.

if (identification_analysis)

options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'MaxFunEvals',6000,'OptimalityTolerance',1e-8,...
 'FunctionTolerance',1e-8);
options1 = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','MaxFunEvals',6000,'OptimalityTolerance',1e-8,...
 'FunctionTolerance',1e-8);

x=(0:.05:2)';
%x=(-.2:.01:0)';
%x=(-4:.1:4)';
n=size(x,1);

if (strcmp(truemixing,'continuous'))

  if (regen_true_model | ~exist('thetatrue'))
    thetatrue(1)=-.5;
    thetatrue(2)=1;
    thetatrue(3)=3;
    thetatrue(4)=1;
    thetatrue(5)=.1;
    thetatrue(6)=.5;
    thetatrue(7)=-1;
    thetatrue(8)=.1;
    thetatrue(9)=-1;
    thetatrue(10)=.7;
    thetatrue(11)=0;
  end

  [trueprob,dtrueprob,true_p,true_nparams]=cprob1(x,thetatrue,T);  % true_nparams is a 4xnt_true_types matrix providing the mean and std deviation of the
                                                                   % mixed normal beta(1) (intercept, with mean and std in rows 1 and 2) and 
                                                                   % beta(2) (slope of x, with mean and std respectively in rows 3 and 4)
                                                                   % where each column is a separate mixture component and true_p is the corresponding
                                                                   % mixing probability for the mixture of bivariate normals. Note that beta(2)
                                                                   % is a marginal distribution that accounts for its covariance with beta(1).
else
  if (regen_true_model | ~exist('thetatrue'))
  thetatrue=randn(n_true_types*2+n_true_types-1,1);
  end
  [trueprob,dtrueprob,htrueprob,true_p,true_betas]=bprob1(x,thetatrue,T);
end

if (strcmp(estmixing,'continuous'))
  theta=randn(5,1);
  kld=@(theta) kldist1(trueprob,x,'cprob',theta,T);
  ssr=@(theta) nlls1(trueprob,x,theta,T); 
else
  if (n_est_types ==1)
  theta=randn(2,1);
  elseif (n_est_types == 2)
  theta=randn(5,1);
  elseif (n_est_types == 3)
  theta=randn(8,1);
  elseif (n_est_types == 4)
  theta=randn(11,1);
  elseif (n_est_types == 5)
  theta=randn(14,1);
  elseif (n_est_types == 6)
  theta=randn(17,1);
  else
  theta=randn(3*n_est_types-1,1);
  end
  kld=@(theta) kldist1(trueprob,x,'bprob',theta,T);
  ssr=@(theta) nlls1(trueprob,x,theta,T); 
end

[thetahat_ssr,ssrmin]=fminunc(ssr,theta,options);
[thetahat_kld,kldmin]=fminunc(kld,theta,options1);

fprintf('estimated and true theta\n');
fprintf('estimated theta (ssr vs kld criterion)\n');
[thetahat_ssr thetahat_kld]
fprintf('true theta\n');
thetatrue'
fprintf('minimum ssr %g\n',ssrmin);
fprintf('minimum kld %g\n',kldmin);

fprintf('characterization of the minimum KL-distance parameter values with ntypes=%i\n',n_est_types);

[kldmin1,dkld,hkld]=kldist1(trueprob,x,'bprob',thetahat_kld,T);
hkld=(hkld+hkld')/2;
fprintf('max absolute gradient of KL distance: %g\n',max(abs(dkld)));
fprintf('condition number of information matrix: %g\n',cond(hkld));
fprintf('Eigenvalues of the information matrix:\n');
eig(hkld)
fprintf('Best approximation parameter values and their asymptotic variances (using pinv)\n');
[thetahat_kld (diag(pinv(hkld)))]

if (strcmp(estmixing,'continuous'))
  if (strcmp(objective,'kldistance'))
    [estprob,destprob,est_p_kld,est_nparams]=cprob1(x,thetahat_kld,1);
  else
    [estprob,destprob,est_p_ssr,est_nparams]=cprob1(x,thetahat_ssr,1);
  end
else
  %[estprob_kld,destprob_kld,hestprob_kld,est_p_kld,est_betas_kld]=bprob1(x,thetahat_kld,1);
  [estprob_kld,destprob_kld,hestprob_kld,est_p_kld,est_betas_kld,dest_p_kld,hest_p_kld,ccps_kld]=bprob1(x,thetahat_kld,1);
  [estprob_ssr,destprob_ssr,hestprob_ssr,est_p_ssr,est_betas_ssr,dest_p_ssr,hest_p_ssr,ccps_ssr]=bprob1(x,thetahat_ssr,1);
  %[estprob_ssr,destprob_ssr,hestprob_ssr,est_p_ssr,est_betas_ssr]=bprob1(x,thetahat_ssr,1);
end


if (T > 1)
 if (strcmp(truemixing,'continuous'))
  [trueprob,dtrueprob,true_p,true_nparams]=cprob1(x,thetatrue,1);
 else
  [trueprob,dtrueprob,htrueprob,true_p,true_betas]=bprob1(x,thetatrue,1);
 end
end

ccp_err_kld=max(max(abs(trueprob-estprob_kld)));
ccp_err_ssr=max(max(abs(trueprob-estprob_ssr)));
fprintf('max absolute difference between true and estimated CCP on the x grid: %g (minimum kl approximation)\n',ccp_err_kld);
fprintf('max absolute difference between true and estimated CCP on the x grid: %g (least squares approximation)\n',ccp_err_ssr);

%fprintf('Product of hessian of KLD and gradient of CCP at best-approximating parameter values\n');
%hkld*destprob_kld'
fprintf('Asymptotic variance of CCP at the x grid via delta theorem (using pseudo-inverse)\n');
[x estprob_kld diag(destprob_kld*pinv(hkld)*destprob_kld')]
fprintf('Asymptotic variance of CCP at the x grid via delta theorem (using inverse)\n');
[x estprob_kld diag(destprob_kld*inv(hkld)*destprob_kld')]

[u,d,v]=svd(hkld);

f1=figure(1);
clf(f1,'reset');
hold on;
plot(x,trueprob,'b-','Linewidth',2);
plot(x,estprob_kld,'r-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: %s heterogeneity, mixture of %i normals',truemixing,n_true_types),sprintf('Estimated: %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
else
legend(sprintf('Truth: %s heterogeneity, %i types',truemixing,n_true_types),sprintf('Estimated: %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
end
xlabel('x');
ylabel('CCP, P(x)');
title({'True vs Approximate P(x)'},{sprintf('Approximate P(x) Kullback-Leibler distance: %g  max absolute error: %g',kldmin,ccp_err_kld)});
hold off;


f2=figure(2);
clf(f2,'reset');
hold on;
plot(x,trueprob,'b-','Linewidth',2);
plot(x,estprob_ssr,'r-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: %s heterogeneity, mixture of %i normals',truemixing,n_true_types),sprintf('Estimated: %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
else
legend(sprintf('Truth: %s heterogeneity, %i types',truemixing,n_true_types),sprintf('Estimated: %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
end
xlabel('x');
ylabel('CCP, P(x)');
title({'True vs Approximate P(x)'},{sprintf('Approximate P(x) L^2 distance: %g  max absolute error: %g',ssrmin,ccp_err_ssr)});
hold off;

f3=figure(3);
clf(f3,'reset');
hold on;
plot(x,trueprob-estprob_kld,'r-','Linewidth',2);
xlabel('x');
ylabel('CCP, P(x)');
title({'True less Approximate CCP P(x)'},{sprintf('Approximate P(x) Kullback-Leibler distance: %g  max absolute error: %g',kldmin,ccp_err_kld)});
hold off;

f4=figure(4);
clf(f4,'reset');
hold on;
plot(x,trueprob-estprob_ssr,'r-','Linewidth',2);
xlabel('x');
ylabel('CCP, P(x)');
title({'True less Approximate CCP P(x)'},{sprintf('Approximate P(x) L^2 distance: %g  max absolute error: %g',ssrmin,ccp_err_ssr)});
hold off;

f5=figure(5);
clf(f5,'reset');
hold on;

est_betas_kld_stacked=reshape(est_betas_kld,2,n_est_types);
est_intercepts_kld=est_betas_kld_stacked(1,:)';
est_slopes_kld=est_betas_kld_stacked(2,:)';

[sb,si]=sort(est_intercepts_kld);

if (strcmp(truemixing,'continuous'))

  if (n_true_types == 1)
  true_intercept_params=true_nparams(1:2);
  xl=true_intercept_params(1)-4*true_intercept_params(2);
  xu=true_intercept_params(1)+4*true_intercept_params(2);
  else
  true_intercept_params=true_nparams(1:2,:);
  xl=min(true_intercept_params(1,:))-4*max(true_intercept_params(2,:));
  xu=max(true_intercept_params(1,:))+4*max(true_intercept_params(2,:));
  end
  xl=min(xl,min(est_intercepts_kld));
  xu=max(xu,max(est_intercepts_kld));
  xu=xu+.1*(xu-xl);
  xl=xl-.1*(xu-xl);
  xbeta=(xl:(xu-xl)/100:xu)';
  intercept_cdf=mixed_normal_cdf(true_p,true_intercept_params,xbeta);
  cdf_err=abs(mixed_normal_cdf(true_p,true_intercept_params,sb(1)));
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(end))-1)];
  plot(xbeta,intercept_cdf,'b-','Linewidth',2);

else

  xl=min([true_betas;est_betas_kld]);
  xu=max([true_betas;est_betas_kld]);
  xu=xu+.1*(xu-xl);
  xl=xl-.1*(xu-xl);
  [tb,ti]=sort(true_betas);
  true_cdf=cumsum(true_p(ti));
  true_cdf_fh=line([xl tb(1)],[0 0],'Color','b','Linestyle','-','Linewidth',2);
  line([tb(1) tb(1)],[0 true_cdf(1)],'Color','b','Linestyle',':','Linewidth',1);
  for i=2:n_true_types
    line([tb(i-1) tb(i)],[true_cdf(i-1) true_cdf(i-1)],'Color','b','Linestyle','-','Linewidth',2);
    line([tb(i) tb(i)],[true_cdf(i-1) true_cdf(i)],'Color','b','Linestyle',':','Linewidth',1);
  end
  line([tb(end) xu],[1 1],'Color','b','Linestyle','-','Linewidth',2);

end

est_cdf=cumsum(est_p_kld(si));
est_cdf_fh=line([xl sb(1)],[0 0],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=abs(mixed_normal_cdf(true_p,true_intercept_params,sb(1)));
else
  cdf_err=abs(discrete_cdf(sb(1),1,tb,true_p));
  cdf_err=[cdf_err; abs(discrete_cdf(sb(1),0,tb,true_p))];
end
line([sb(1) sb(1)],[0 est_cdf(1)],'Color','r','Linestyle',':','Linewidth',1);
for i=2:n_est_types
  line([sb(i-1) sb(i)],[est_cdf(i-1) est_cdf(i-1)],'Color','r','Linestyle','-','Linewidth',2);
  line([sb(i) sb(i)],[est_cdf(i-1) est_cdf(i)],'Color','r','Linestyle',':','Linewidth',1);
  if (strcmp(truemixing,'continuous'))
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(i-1))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(i))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(i))-est_cdf(i))];
  else
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i))];
  end
end
line([sb(end) xu],[1 1],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(end))-1)];
else
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),1,tb,true_p)-1)];
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),0,tb,true_p)-1)];
end

if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: Mixture of %i normals',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Northwest');
else
legend([true_cdf_fh est_cdf_fh],sprintf('Truth: finite mixture with %i types',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Northwest');
end
xlabel('\beta');
ylabel('Cumulative Distribution Function');
title({'True vs approximate CDF of intercept: minimum Kullback-Leibler distance'},{sprintf('Maximum absolute error between true and approximate CDF: %g',max(cdf_err))});
axis('tight');
hold off;

f6=figure(6);
clf(f6,'reset');
hold on;

[sb,si]=sort(est_slopes_kld);

if (strcmp(truemixing,'continuous'))

  if (n_true_types == 1)
  true_slope_params=true_nparams(3:4);
  true_slope_params(2)=sqrt((true_nparams(5)^2)*true_nparams(2)^2+true_nparams(4)^2);
  xl=true_slope_params(1)-4*true_slope_params(2);
  xu=true_slope_params(1)+4*true_slope_params(2);
  else
  true_slope_params=true_nparams(3:4,:);
  true_slope_params(2,:)=sqrt((true_nparams(5,:).^2).*(true_nparams(2,:).^2)+true_nparams(4,:).^2);
  xl=min(true_slope_params(1,:))-4*max(true_slope_params(2,:));
  xu=max(true_slope_params(1,:))+4*max(true_slope_params(2,:));
  end
  xl=min(xl,min(est_slopes_kld));
  xu=max(xu,max(est_slopes_kld));
  xu=xu+.1*(xu-xl);
  xl=xl-.1*(xu-xl);
  xbeta=(xl:(xu-xl)/100:xu)';
  slope_cdf=mixed_normal_cdf(true_p,true_slope_params,xbeta);
  cdf_err=abs(mixed_normal_cdf(true_p,true_slope_params,sb(1)));
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(end))-1)];
  plot(xbeta,slope_cdf,'b-','Linewidth',2);

else

  xl=min([true_betas;est_betas_kld]);
  xu=max([true_betas;est_betas_kld]);
  xu=xu+.1*(xu-xl);
  xl=xl-.1*(xu-xl);
  [tb,ti]=sort(true_betas);
  true_cdf=cumsum(true_p(ti));
  true_cdf_fh=line([xl tb(1)],[0 0],'Color','b','Linestyle','-','Linewidth',2);
  line([tb(1) tb(1)],[0 true_cdf(1)],'Color','b','Linestyle',':','Linewidth',1);
  for i=2:n_true_types
    line([tb(i-1) tb(i)],[true_cdf(i-1) true_cdf(i-1)],'Color','b','Linestyle','-','Linewidth',2);
    line([tb(i) tb(i)],[true_cdf(i-1) true_cdf(i)],'Color','b','Linestyle',':','Linewidth',1);
  end
  line([tb(end) xu],[1 1],'Color','b','Linestyle','-','Linewidth',2);

end

est_cdf=cumsum(est_p_kld(si));
est_cdf_fh=line([xl sb(1)],[0 0],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=abs(mixed_normal_cdf(true_p,true_slope_params,sb(1)));
else
  cdf_err=abs(discrete_cdf(sb(1),1,tb,true_p));
  cdf_err=[cdf_err; abs(discrete_cdf(sb(1),0,tb,true_p))];
end
line([sb(1) sb(1)],[0 est_cdf(1)],'Color','r','Linestyle',':','Linewidth',1);
for i=2:n_est_types
  line([sb(i-1) sb(i)],[est_cdf(i-1) est_cdf(i-1)],'Color','r','Linestyle','-','Linewidth',2);
  line([sb(i) sb(i)],[est_cdf(i-1) est_cdf(i)],'Color','r','Linestyle',':','Linewidth',1);
  if (strcmp(truemixing,'continuous'))
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(i-1))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(i))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(i))-est_cdf(i))];
  else
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i))];
  end
end
line([sb(end) xu],[1 1],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(end))-1)];
else
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),1,tb,true_p)-1)];
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),0,tb,true_p)-1)];
end

if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: Mixture of %i normals',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Northwest');
else
legend([true_cdf_fh est_cdf_fh],sprintf('Truth: finite mixture with %i types',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Northwest');
end
xlabel('\beta');
ylabel('Cumulative Distribution Function');
title({'True vs approximate CDF of x coefficient: minimum Kullback-Leibler distance'},{sprintf('Maximum absolute error between true and approximate CDF: %g',max(cdf_err))});
axis('tight');
hold off;

if (n_est_types > 1)

f7=figure(7);
clf(f7,'reset');
hold on;
plot(x,trueprob,'b-','Linewidth',2);
plot(x,estprob_kld,'r-','Linewidth',2);
plot(x,ccps_kld,'k--','Linewidth',1);
if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: %s heterogeneity, mixture of %i normals',truemixing,n_true_types),sprintf('Estimated: %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
else
legend(sprintf('Truth: %s heterogeneity, %i types',truemixing,n_true_types),sprintf('Estimated: %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
end
xlabel('x');
ylabel('CCP, P(x)');
title({'True vs Approximate P(x)'},{sprintf('Approximate P(x) Kullback-Leibler distance: %g  max absolute error: %g',kldmin,ccp_err_kld)});
hold off;

fprintf('Type probabilities\n');
[(1:n_est_types)' est_p_kld]

end

moments_true=struct;
moments_est=struct;

if (strcmp(truemixing,'continuous'))
   moments_true.intercept_mean=0; 
   for i=1:n_true_types
     moments_true.intercept_mean=moments_true.intercept_mean+true_p(i)*true_intercept_params(1,i); 
   end
   moments_true.slope_mean=0; 
   for i=1:n_true_types
     moments_true.slope_mean=moments_true.slope_mean+true_p(i)*true_slope_params(1,i); 
   end
   moments_true.intercept_m2=0; 
   for i=1:n_true_types
     moments_true.intercept_m2=moments_true.intercept_m2+true_p(i)*(true_intercept_params(2,i)^2+(true_intercept_params(1,i)^2)); 
   end
   moments_true.slope_m2=0; 
   for i=1:n_true_types
     moments_true.slope_m2=moments_true.slope_m2+true_p(i)*(true_slope_params(2,i)^2+(true_slope_params(1,i)^2)); 
   end
   moments_true.intercept_std=sqrt(moments_true.intercept_m2-moments_true.intercept_mean^2);
   moments_true.slope_std=sqrt(moments_true.slope_m2-moments_true.slope_mean^2);

   moments_true.xm=0; 
   for i=1:n_true_types
     moments_true.xm=moments_true.xm+true_p(i)*(true_nparams(5,i)*(true_nparams(2,i)^2)+(true_nparams(1,i)*true_nparams(3,i))); 
   end

   moments_true.intercept_slope_corr=moments_true.xm-moments_true.intercept_mean*moments_true.slope_mean;
   moments_true.intercept_slope_corr=moments_true.intercept_slope_corr/(moments_true.intercept_std*moments_true.slope_std);

end

if (strcmp(estmixing,'discrete'))
   moments_est.intercept_mean=0; 
   for i=1:n_est_types
     moments_est.intercept_mean=moments_est.intercept_mean+est_p_kld(i)*est_intercepts_kld(i); 
   end
   moments_est.slope_mean=0; 
   for i=1:n_est_types
     moments_est.slope_mean=moments_est.slope_mean+est_p_kld(i)*est_slopes_kld(i); 
   end
   moments_est.intercept_m2=0; 
   for i=1:n_est_types
     moments_est.intercept_m2=moments_est.intercept_m2+est_p_kld(i)*(est_intercepts_kld(i)^2); 
   end
   moments_est.slope_m2=0; 
   for i=1:n_est_types
     moments_est.slope_m2=moments_est.slope_m2+est_p_kld(i)*(est_slopes_kld(i)^2); 
   end
   moments_est.intercept_std=sqrt(moments_est.intercept_m2-moments_est.intercept_mean^2);
   moments_est.slope_std=sqrt(moments_est.slope_m2-moments_est.slope_mean^2);

   moments_est.xm=0;
   for i=1:n_est_types
     moments_est.xm=moments_est.xm+est_p_kld(i)*(est_slopes_kld(i)*est_intercepts_kld(i)); 
   end

   moments_est.intercept_slope_corr=moments_est.xm-moments_est.intercept_mean*moments_est.slope_mean; 
   moments_est.intercept_slope_corr=moments_est.intercept_slope_corr/(moments_est.intercept_std*moments_est.slope_std);
  
end

fprintf('Comparison of true vs estimated moments of intercept/slope random coefficients\n');
fprintf('Moment                       True                Approximated\n');
fprintf('Intercept mean               %g                  %g\n',moments_true.intercept_mean,moments_est.intercept_mean);
fprintf('x Coefficient mean           %g                  %g\n',moments_true.slope_mean,moments_est.slope_mean);
fprintf('Intercept std                %g                  %g\n',moments_true.intercept_std,moments_est.intercept_std);
fprintf('x Coefficient std            %g                  %g\n',moments_true.slope_std,moments_est.slope_std);
fprintf('corr(Intercept,slope)        %g                  %g\n',moments_true.intercept_slope_corr,moments_est.intercept_slope_corr);

end

if (ml_estimation)

   x=(0:.05:2)';

  if (regen_true_model | ~exist('thetatrue'))
    thetatrue(1)=-.5;
    thetatrue(2)=1;
    thetatrue(3)=3;
    thetatrue(4)=1;
    thetatrue(5)=.1;
    thetatrue(6)=.5;
    thetatrue(7)=-1;
    thetatrue(8)=.1;
    thetatrue(9)=-1;
    thetatrue(10)=.7;
    thetatrue(11)=0;
  end

  [trueprob,dtrueprob,true_p,true_nparams]=cprob1(0,thetatrue,1);  % true_nparams is a 4xnt_true_types matrix providing the mean and std deviation of the
                                                                   % mixed normal beta(1) (intercept, with mean and std in rows 1 and 2) and 
                                                                   % beta(2) (slope of x, with mean and std respectively in rows 3 and 4)
                                                                   % where each column is a separate mixture component and true_p is the corresponding
                                                                   % mixing probability for the mixture of bivariate normals. Note that beta(2)
                                                                   % is a marginal distribution that accounts for its covariance with beta(1).

  if (gendat)

    [ydata,xdata,tv]=gendata(nobs,T,thetatrue,true_p,truemixing);
    save('xdata','xdata');
    save('ydata','ydata');
    save('tv','tv');

  else

    if (~exist('xdata'))
       xdata=load('xdata');
       ydata=load('ydata');
       tv=load('tv');
       if ((size(xdata,1) ~= size(ydata,1))|(size(xdata,2) ~= size(ydata,2)))
          fprintf('Error: sizes of x and y are not the same, regenerating new data\n');
          [ydata,xdata]=gendata(nobs,T,thetatrue,true_p,truemixing);
       end
    else
       if ((size(xdata,1) ~= size(ydata,1))|(size(xdata,2) ~= size(ydata,2)))
          fprintf('Error: sizes of x and y are not the same, regenerating new data\n');
          [ydata,xdata]=gendata(nobs,T,thetatrue,true_p,truemixing);
       end
    end

  end

  % regenerate the true mixed choice probabilities for the actual covariate vector and true parameter vector

  [trueprob,dtrueprob,true_p,true_nparams]=cprob1(x,thetatrue,1);  % true_nparams is a 4xnt_true_types matrix providing the mean and std deviation of the
                                                                   % mixed normal beta(1) (intercept, with mean and std in rows 1 and 2) and 
                                                                   % beta(2) (slope of x, with mean and std respectively in rows 3 and 4)
                                                                   % where each column is a separate mixture component and true_p is the corresponding
                                                                   % mixing probability for the mixture of bivariate normals. Note that beta(2)
                                                                   % is a marginal distribution that accounts for its covariance with beta(1).
  theta=randn(3*n_est_types-1,1);

  if (newstart)
     thetastart=theta;
  else
     if (~exist('thetastart'))
       thetastart=theta;
     else
       if (numel(thetastart) ~= numel(theta))
       thetastart=theta;
       end
     end
  end

  if (T == 1)

     if (strcmp(probtype,'bprob1'))

       lf=@(theta) nllf1(ydata,xdata,theta);
 
     else

       lf=@(theta) nllf2(ydata,xdata,theta);

     end

  else % in the panel case, always use bprob2 and nllf2

    lf=@(theta) nllf2(ydata,xdata,theta);

  end

  options=optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','MaxFunEvals',6000,'OptimalityTolerance',1e-8,...
 'FunctionTolerance',1e-8);
  [thetahat,lfmax,]=fminunc(lf,thetastart,options);
  lfmax=-lfmax;

  % compute standard errors as inverse of information matrix

  if (T > 1)
     [lfv,dllf,hllf,im]=nllf2(ydata,xdata,thetahat);
  else
     if (strcmp(probtype,'bprob1'))
        [lfv,dllf,hllf,im]=nllf1(ydata,xdata,thetahat);
     else
        [lfv,dllf,hllf,im]=nllf2(ydata,xdata,thetahat);
     end
  end
  covm=inv(hllf)*im*inv(hllf);
  stderr=sqrt(diag(covm));

  if (strcmp(estmixing,'continuous'))
     fprintf('Optimized value of the log-likelihood for %s normal(mu,sigma^2) specification of random coefficients: %g\n',estmixing,lfmax);
  else
     fprintf('Optimized value of the log-likelihood for %s mixture specification of random coefficients with %i discrete types: %g\n',...
     estmixing,n_est_types,lfmax);
  end
  fprintf('Number of observations: %i\n',nobs);
  fprintf('AIC: %g  BIC: %g\n',-2*lfmax+2*size(thetahat,1),-2*lfmax+log(nobs)*size(thetahat,1));
  fprintf('\nEstimated parameter vector and standard errors\n');
  [thetahat stderr]


  if (strcmp(truemixing,'continuous'));
   fprintf('True model is a mixture of normals with %i mixture components\n',n_true_types);
  else
   fprintf('True model has a discrete distribution over beta with %i types\n',n_true_types);
  end

  if (strcmp(estmixing,'continuous'));
     estprob=cprob1(x,thetahat,1);
  else
     [estprob,destprob,hestprob,est_p,est_betas,dest_p]=bprob1(x,thetahat,1);
  end

  if (strcmp(truemixing,'continuous'))
    trueprob=cprob1(x,thetatrue,1);
  else
    trueprob=bprob1(x,thetatrue,1);
  end

  if (n_est_types > 1);
    var_p=dest_p*covm(2*n_est_types+1:end,2*n_est_types+1:end)*(dest_p');
    std_err_p=sqrt(diag(var_p));
    for i=1:n_est_types
      fprintf('Estimated probability of type %i: %g %g\n',i,est_p(i),std_err_p(i));
    end
  end

  fprintf('\nEstimated CCPs and std errors at various x values\n');
  [estprob_x,destprob_x]=bprob1(x,thetahat,1);
  covm_p=destprob_x*covm*(destprob_x');
  std_err_px=sqrt(diag(covm_p));
  for i=1:numel(x)
      fprintf('P(%g)=%g std err=%g\n',x(i),estprob_x(i),std_err_px(i));
  end

  f1=figure(1);
  clf(f1,'reset');
  hold on;
  plot(x,estprob,'r-','Linewidth',2);
  plot(x,trueprob,'b-','Linewidth',2);
  xlabel('x');
  ylabel('Estimated CCP P(x)');
  legend('Estimated CCP P(x)','True CCP P(x)','Location','Best');
  if (strcmp(estmixing,'continuous'))
    title({'Maximum likelihood estimate of P(x) using N(\mu,\sigma^2) distribution',sprintf('max difference between true and estimated P(x): %g',max(abs(trueprob-estprob)))});
  else
    title({sprintf('Maximum likelihood estimate of P(x) using discrete mixture with %i types',n_est_types),sprintf('max difference between true and estimated P(x): %g',max(abs(trueprob-estprob)))});
  end
  hold off;

f2=figure(2);
clf(f2,'reset');
hold on;

est_betas_stacked=reshape(est_betas,2,n_est_types);
est_intercepts=est_betas_stacked(1,:)';
est_slopes=est_betas_stacked(2,:)';

[sb,si]=sort(est_intercepts);

if (strcmp(truemixing,'continuous'))

  if (n_true_types == 1)
  true_intercept_params=true_nparams(1:2);
  xl=true_intercept_params(1)-4*true_intercept_params(2);
  xu=true_intercept_params(1)+4*true_intercept_params(2);
  else
  true_intercept_params=true_nparams(1:2,:);
  xl=min(true_intercept_params(1,:))-4*max(true_intercept_params(2,:));
  xu=max(true_intercept_params(1,:))+4*max(true_intercept_params(2,:));
  end
  xl=min(xl,min(est_intercepts));
  xu=max(xu,max(est_intercepts));
  xu=xu+.1*(xu-xl);
  xl=xl-.1*(xu-xl);
  xbeta=(xl:(xu-xl)/100:xu)';
  intercept_cdf=mixed_normal_cdf(true_p,true_intercept_params,xbeta);
  cdf_err=abs(mixed_normal_cdf(true_p,true_intercept_params,sb(1)));
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(end))-1)];
  plot(xbeta,intercept_cdf,'b-','Linewidth',2);

else

  xl=min([true_betas;est_betas_kld]);
  xu=max([true_betas;est_betas_kld]);
  xu=xu+.1*(xu-xl);
  xl=xl-.1*(xu-xl);
  [tb,ti]=sort(true_betas);
  true_cdf=cumsum(true_p(ti));
  true_cdf_fh=line([xl tb(1)],[0 0],'Color','b','Linestyle','-','Linewidth',2);
  line([tb(1) tb(1)],[0 true_cdf(1)],'Color','b','Linestyle',':','Linewidth',1);
  for i=2:n_true_types
    line([tb(i-1) tb(i)],[true_cdf(i-1) true_cdf(i-1)],'Color','b','Linestyle','-','Linewidth',2);
    line([tb(i) tb(i)],[true_cdf(i-1) true_cdf(i)],'Color','b','Linestyle',':','Linewidth',1);
  end
  line([tb(end) xu],[1 1],'Color','b','Linestyle','-','Linewidth',2);

end

est_cdf=cumsum(est_p(si));
est_cdf_fh=line([xl sb(1)],[0 0],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=abs(mixed_normal_cdf(true_p,true_intercept_params,sb(1)));
else
  cdf_err=abs(discrete_cdf(sb(1),1,tb,true_p));
  cdf_err=[cdf_err; abs(discrete_cdf(sb(1),0,tb,true_p))];
end
line([sb(1) sb(1)],[0 est_cdf(1)],'Color','r','Linestyle',':','Linewidth',1);
for i=2:n_est_types
  line([sb(i-1) sb(i)],[est_cdf(i-1) est_cdf(i-1)],'Color','r','Linestyle','-','Linewidth',2);
  line([sb(i) sb(i)],[est_cdf(i-1) est_cdf(i)],'Color','r','Linestyle',':','Linewidth',1);
  if (strcmp(truemixing,'continuous'))
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(i-1))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(i))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(i))-est_cdf(i))];
  else
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i))];
  end
end
line([sb(end) xu],[1 1],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_intercept_params,sb(end))-1)];
else
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),1,tb,true_p)-1)];
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),0,tb,true_p)-1)];
end

if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: Mixture of %i normals',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Northwest');
else
legend([true_cdf_fh est_cdf_fh],sprintf('Truth: finite mixture with %i types',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Northwest');
end
xlabel('\beta');
ylabel('Cumulative Distribution Function');
title({'True vs approximate CDF of intercept: minimum Kullback-Leibler distance'},{sprintf('Maximum absolute error between true and approximate CDF: %g',max(cdf_err))});
axis('tight');
hold off;

f3=figure(3);
clf(f3,'reset');
hold on;

[sb,si]=sort(est_slopes);

if (strcmp(truemixing,'continuous'))

  if (n_true_types == 1)
  true_slope_params=true_nparams(3:4);
  true_slope_params(2)=sqrt((true_nparams(5)^2)*true_nparams(2)^2+true_nparams(4)^2);
  xl=true_slope_params(1)-4*true_slope_params(2);
  xu=true_slope_params(1)+4*true_slope_params(2);
  else
  true_slope_params=true_nparams(3:4,:);
  true_slope_params(2,:)=sqrt((true_nparams(5,:).^2).*(true_nparams(2,:).^2)+true_nparams(4,:).^2);
  xl=min(true_slope_params(1,:))-4*max(true_slope_params(2,:));
  xu=max(true_slope_params(1,:))+4*max(true_slope_params(2,:));
  end
  xl=min(xl,min(est_slopes));
  xu=max(xu,max(est_slopes));
  xu=xu+.1*(xu-xl);
  xl=xl-.1*(xu-xl);
  xbeta=(xl:(xu-xl)/100:xu)';
  slope_cdf=mixed_normal_cdf(true_p,true_slope_params,xbeta);
  cdf_err=abs(mixed_normal_cdf(true_p,true_slope_params,sb(1)));
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(end))-1)];
  plot(xbeta,slope_cdf,'b-','Linewidth',2);

else

  xl=min([true_betas;est_betas]);
  xu=max([true_betas;est_betas]);
  xu=xu+.1*(xu-xl);
  xl=xl-.1*(xu-xl);
  [tb,ti]=sort(true_betas);
  true_cdf=cumsum(true_p(ti));
  true_cdf_fh=line([xl tb(1)],[0 0],'Color','b','Linestyle','-','Linewidth',2);
  line([tb(1) tb(1)],[0 true_cdf(1)],'Color','b','Linestyle',':','Linewidth',1);
  for i=2:n_true_types
    line([tb(i-1) tb(i)],[true_cdf(i-1) true_cdf(i-1)],'Color','b','Linestyle','-','Linewidth',2);
    line([tb(i) tb(i)],[true_cdf(i-1) true_cdf(i)],'Color','b','Linestyle',':','Linewidth',1);
  end
  line([tb(end) xu],[1 1],'Color','b','Linestyle','-','Linewidth',2);

end

est_cdf=cumsum(est_p(si));
est_cdf_fh=line([xl sb(1)],[0 0],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=abs(mixed_normal_cdf(true_p,true_slope_params,sb(1)));
else
  cdf_err=abs(discrete_cdf(sb(1),1,tb,true_p));
  cdf_err=[cdf_err; abs(discrete_cdf(sb(1),0,tb,true_p))];
end
line([sb(1) sb(1)],[0 est_cdf(1)],'Color','r','Linestyle',':','Linewidth',1);
for i=2:n_est_types
  line([sb(i-1) sb(i)],[est_cdf(i-1) est_cdf(i-1)],'Color','r','Linestyle','-','Linewidth',2);
  line([sb(i) sb(i)],[est_cdf(i-1) est_cdf(i)],'Color','r','Linestyle',':','Linewidth',1);
  if (strcmp(truemixing,'continuous'))
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(i-1))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(i))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(i))-est_cdf(i))];
  else
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i))];
  end
end
line([sb(end) xu],[1 1],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_slope_params,sb(end))-1)];
else
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),1,tb,true_p)-1)];
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),0,tb,true_p)-1)];
end

if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: Mixture of %i normals',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Northwest');
else
legend([true_cdf_fh est_cdf_fh],sprintf('Truth: finite mixture with %i types',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Northwest');
end
xlabel('\beta');
ylabel('Cumulative Distribution Function');
title({'True vs approximate CDF of x coefficient: minimum Kullback-Leibler distance'},{sprintf('Maximum absolute error between true and approximate CDF: %g',max(cdf_err))});
axis('tight');
hold off;

if (n_est_types > 1)

  est_ccps=zeros(numel(x),n_est_types);

  for i=1:n_est_types

    est_ccps(:,i)=1./(1+exp(est_betas_stacked(1,i)+x*est_betas_stacked(2,i)));

  end

f4=figure(4);
clf(f4,'reset');
hold on;
plot(x,trueprob,'b-','Linewidth',2);
plot(x,estprob,'r-','Linewidth',2);
plot(x,est_ccps,'k--','Linewidth',1);
if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: %s heterogeneity, mixture of %i normals',truemixing,n_true_types),sprintf('Estimated: %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
else
legend(sprintf('Truth: %s heterogeneity, %i types',truemixing,n_true_types),sprintf('Estimated: %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
end
xlabel('x');
ylabel('CCP, P(x)');
title(sprintf('True vs estimated CCPs and type-specific CCPs, %i types',n_est_types));
hold off;

end

moments_true=struct;
moments_est=struct;

if (strcmp(truemixing,'continuous'))
   moments_true.intercept_mean=0; 
   for i=1:n_true_types
     moments_true.intercept_mean=moments_true.intercept_mean+true_p(i)*true_intercept_params(1,i); 
   end
   moments_true.slope_mean=0; 
   for i=1:n_true_types
     moments_true.slope_mean=moments_true.slope_mean+true_p(i)*true_slope_params(1,i); 
   end
   moments_true.intercept_m2=0; 
   for i=1:n_true_types
     moments_true.intercept_m2=moments_true.intercept_m2+true_p(i)*(true_intercept_params(2,i)^2+(true_intercept_params(1,i)^2)); 
   end
   moments_true.slope_m2=0; 
   for i=1:n_true_types
     moments_true.slope_m2=moments_true.slope_m2+true_p(i)*(true_slope_params(2,i)^2+(true_slope_params(1,i)^2)); 
   end
   moments_true.intercept_std=sqrt(moments_true.intercept_m2-moments_true.intercept_mean^2);
   moments_true.slope_std=sqrt(moments_true.slope_m2-moments_true.slope_mean^2);

   moments_true.xm=0; 
   for i=1:n_true_types
     moments_true.xm=moments_true.xm+true_p(i)*(true_nparams(5,i)*(true_nparams(2,i)^2)+(true_nparams(1,i)*true_nparams(3,i))); 
   end

   moments_true.intercept_slope_corr=moments_true.xm-moments_true.intercept_mean*moments_true.slope_mean;
   moments_true.intercept_slope_corr=moments_true.intercept_slope_corr/(moments_true.intercept_std*moments_true.slope_std);

end

if (strcmp(estmixing,'discrete'))
   moments_est.intercept_mean=0; 
   for i=1:n_est_types
     moments_est.intercept_mean=moments_est.intercept_mean+est_p(i)*est_intercepts(i); 
   end
   moments_est.slope_mean=0; 
   for i=1:n_est_types
     moments_est.slope_mean=moments_est.slope_mean+est_p(i)*est_slopes(i); 
   end
   moments_est.intercept_m2=0; 
   for i=1:n_est_types
     moments_est.intercept_m2=moments_est.intercept_m2+est_p(i)*(est_intercepts(i)^2); 
   end
   moments_est.slope_m2=0; 
   for i=1:n_est_types
     moments_est.slope_m2=moments_est.slope_m2+est_p(i)*(est_slopes(i)^2); 
   end
   moments_est.intercept_std=sqrt(moments_est.intercept_m2-moments_est.intercept_mean^2);
   moments_est.slope_std=sqrt(moments_est.slope_m2-moments_est.slope_mean^2);

   moments_est.xm=0;
   for i=1:n_est_types
     moments_est.xm=moments_est.xm+est_p(i)*(est_slopes(i)*est_intercepts(i)); 
   end

   moments_est.intercept_slope_corr=moments_est.xm-moments_est.intercept_mean*moments_est.slope_mean; 
   moments_est.intercept_slope_corr=moments_est.intercept_slope_corr/(moments_est.intercept_std*moments_est.slope_std);
  
end

fprintf('Comparison of true vs estimated moments of intercept/slope random coefficients\n');
fprintf('Moment                       True                Approximated\n');
fprintf('Intercept mean               %g                  %g\n',moments_true.intercept_mean,moments_est.intercept_mean);
fprintf('x Coefficient mean           %g                  %g\n',moments_true.slope_mean,moments_est.slope_mean);
fprintf('Intercept std                %g                  %g\n',moments_true.intercept_std,moments_est.intercept_std);
fprintf('x Coefficient std            %g                  %g\n',moments_true.slope_std,moments_est.slope_std);
fprintf('corr(Intercept,slope)        %g                  %g\n',moments_true.intercept_slope_corr,moments_est.intercept_slope_corr);

end
