% mixed_logit_estimator: estimation/identification of underling distribution of random coefficients
%                        of a simple example binary logit model under alternative distributions for
%                        unobserved heterogeneity
%
%                        John Rust, July 2024


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
                            % specification with up to 5 points of support) by maximum likelihood
regen_true_model=0;         % Enter 1 to randomly draw parameters for the "true model" of unobserved types/random coefficients,
                            % otherwise the previously drawn values (stored in thetatrue vector) will be used if present in memory.

objective='kldistance';     % when identification_analysis=1, this specifies the objective to be minimized to uncover the distribution
                            % of random coefficients using a multinomial distribution with k "types" of individuals (i.e. a single layer neural network
                            % with a logistic or softmax transfer function and k hidden units). 
                            % set objective='kldistance' to minimize the Kullback-Leibler distance between the true CCP (with continuously
                            %       distributed coefficients, potentially) and the approximate CCP using the finite mixture of types
                            % set objective='leastsquares' to minimize the squared error prediction error between the true CCP and approximate one

truemixing='discrete';    % a flag to specify the true mixing distribution used to generate P(x): continuous vs discrete
estmixing='discrete';       % a flag to specify what type of model to estimate for F(beta): continuyous vs discrete

T=1;                       % length of panel. Set T=1 for pure cross sectional data.

n_est_types=1;             % if estmixing='discrete' then this gives the number of discrete mass points in the estimated discrete F(beta) distribution
                            % if estmixing='continuous' and n_est_types > 1, then a mixture of n_est_types normals is estimated
n_true_types=1;            % if truemixing='discrete' then this gives the number of discrete mass points for F(beta) used to
                            % calculate true choice probability P(x). If truemixing or estmixing is continuous then if n_true_types=1
                            % 3 parameters are estimated: theta(1)=alpha (the constant term in the logit) and mu=theta(2) and
                            % sigma=exp(theta(3)), there (mu,sigma^2) are the mean and variance of a continuous normally
                            % distributed specification for F(beta).
                            %
                            % If n_true_types > 1 then a mixture of normals is used for the mixing distribution F(beta) where
                            % theta(1) is the common intercept in the binomial logits, and (theta(2),...,theta(1+2*n_true_types)) 
                            % are pairs of parameters for (mu(t),sigma(t)), t=1,...,n_true_types where mu(t)=theta(1+2*t) and sigma(t)=exp(theta(2+2*t))
                            % and parameters (theta(1+2*n_true_types+1),....,theta(3*n_true_types)) are parameters that give the 
                            % probabilities of the n_true_types normal component distributions via a multinomial logit parameterization
                            % where p(1)=1/(1+exp(theta(1+2*n_true_types+1)))+ ... + exp(theta(3*n_true_types)))

if (identification_analysis)

options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'MaxFunEvals',6000,'OptimalityTolerance',1e-8,...
 'FunctionTolerance',1e-8);

x=(-1:.1:1)';
%x=(-.2:.01:0)';
%x=(-4:.1:4)';
n=size(x,1);

if (strcmp(truemixing,'continuous'))
  if (regen_true_model | ~exist('thetatrue'))
  thetatrue=randn(n_true_types*3,1);
  thetatrue(2)=-3;
  thetatrue(3)=-3;
  thetatrue(4)=3;
  thetatrue(5)=.1;
  thetatrue(6)=0;
  thetatrue(7)=2;
  end
  [trueprob,dtrueprob,true_p,true_nparams]=cprob(x,thetatrue,T);
else
  if (regen_true_model | ~exist('thetatrue'))
  thetatrue=randn(n_true_types*2,1);
  end
  [trueprob,dtrueprob,true_p,true_betas]=bprob(x,thetatrue,T);
end

if (strcmp(estmixing,'continuous'))
  theta=randn(3,1);
  kld=@(theta) kldist(trueprob,x,'cprob',theta,T);
  ssr=@(theta) nlls(trueprob,x,theta,T); 
else
  if (n_est_types ==1)
  theta=randn(2,1);
  elseif (n_est_types == 2)
  theta=randn(4,1);
  elseif (n_est_types == 3)
  theta=randn(6,1);
  elseif (n_est_types == 4)
  theta=randn(8,1);
  elseif (n_est_types == 5)
  theta=randn(10,1);
  elseif (n_est_types == 6)
  theta=randn(12,1);
  else
  theta=randn(2*n_est_types,1);
  end
  kld=@(theta) kldist(trueprob,x,'bprob',theta,T);
  ssr=@(theta) nlls(trueprob,x,theta,T); 
end

[thetahat_ssr,ssrmin]=fminunc(ssr,theta,options);
[thetahat_kld,kldmin]=fminunc(kld,theta,options);

fprintf('estimated and true theta\n');
fprintf('estimated theta (ssr vs kld criterion)\n');
[thetahat_ssr thetahat_kld]
fprintf('true theta\n');
thetatrue'
fprintf('minimum ssr %g\n',ssrmin);
fprintf('minimum kld %g\n',kldmin);

fprintf('characterization of the minimum KL-distance parameter values with ntypes=%i\n',n_est_types);

[kldmin1,dkld,hkld]=kldist(trueprob,x,'bprob',thetahat_kld,T);
hkld=(hkld+hkld')/2;
fprintf('max absolute gradient of KL distance: %g\n',max(abs(dkld)));
fprintf('condition number of information matrix: %g\n',cond(hkld));
fprintf('Eigenvalues of the information matrix:\n');
eig(hkld)
fprintf('Best approximation parameter values and their asymptotic variances (using pinv)\n');
[thetahat_kld (diag(pinv(hkld)))]

if (strcmp(estmixing,'continuous'))
  if (strcmp(objective,'kldistance'))
    [estprob,destprob,est_p_kld,est_nparams]=cprob(x,thetahat_kld,1);
  else
    [estprob,destprob,est_p_ssr,est_nparams]=cprob(x,thetahat_ssr,1);
  end
else
  [estprob_kld,destprob_kld,est_p_kld,est_betas_kld]=bprob(x,thetahat_kld,1);
  [estprob_ssr,destprob_ssr,est_p_ssr,est_betas_ssr]=bprob(x,thetahat_ssr,1);
end


if (T > 1)
 if (strcmp(truemixing,'continuous'))
  [trueprob,dtrueprob,true_p,true_nparams]=cprob(x,thetatrue,1);
 else
  [trueprob,dtrueprob,true_p,true_betas]=bprob(x,thetatrue,1);
 end
end

ccp_err_kld=max(abs(trueprob-estprob_kld));
ccp_err_ssr=max(abs(trueprob-estprob_ssr));
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

[sb,si]=sort(est_betas_kld);

if (strcmp(truemixing,'continuous'))

  true_beta_mean=true_nparams(1,:)*true_p;
  true_beta_variance=(true_nparams(2,:).^2)*true_p;
  xl=true_beta_mean-4*sqrt(true_beta_variance);
  xu=true_beta_mean+4*sqrt(true_beta_variance);
  xbeta=(xl:(xu-xl)/100:xu)';
  mixed_cdf=mixed_normal_cdf(true_p,true_nparams,xbeta);
  cdf_err=abs(mixed_normal_cdf(true_p,true_nparams,sb(1)));
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(end))-1)];
  plot(xbeta,mixed_cdf,'b-','Linewidth',2);

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
  cdf_err=abs(mixed_normal_cdf(true_p,true_nparams,sb(1)));
else
  cdf_err=abs(discrete_cdf(sb(1),1,tb,true_p));
  cdf_err=[cdf_err; abs(discrete_cdf(sb(1),0,tb,true_p))];
end
line([sb(1) sb(1)],[0 est_cdf(1)],'Color','r','Linestyle',':','Linewidth',1);
for i=2:n_est_types
  line([sb(i-1) sb(i)],[est_cdf(i-1) est_cdf(i-1)],'Color','r','Linestyle','-','Linewidth',2);
  line([sb(i) sb(i)],[est_cdf(i-1) est_cdf(i)],'Color','r','Linestyle',':','Linewidth',1);
  if (strcmp(truemixing,'continuous'))
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(i-1))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(i))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(i))-est_cdf(i))];
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
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(end))-1)];
else
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),1,tb,true_p)-1)];
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),0,tb,true_p)-1)];
end

if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: Mixture of %i normals',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Southeast');
else
legend([true_cdf_fh est_cdf_fh],sprintf('Truth: finite  mixture with %i types',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Southeast');
end
xlabel('\beta');
ylabel('Cumulative Distribution Function');
title({'True vs approximate CDF of random coefficients: minimum Kullback-Leibler distance'},{sprintf('Maximum absolute error between true and approximate CDF: %g',max(cdf_err))});
axis('tight');
hold off;

f4=figure(4);
clf(f4,'reset');
hold on;
if (strcmp(truemixing,'continuous'))
  plot(xbeta,mixed_cdf,'b-','Linewidth',2);
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

[sb,si]=sort(est_betas_ssr);
est_cdf=cumsum(est_p_ssr(si));
line([xl sb(1)],[0 0],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=abs(mixed_normal_cdf(true_p,true_nparams,sb(1)));
else
  cdf_err=abs(discrete_cdf(sb(1),1,tb,true_p));
  cdf_err=[cdf_err; abs(discrete_cdf(sb(1),0,tb,true_p))];
end
line([sb(1) sb(1)],[0 est_cdf(1)],'Color','r','Linestyle',':','Linewidth',1);
for i=2:n_est_types
  line([sb(i-1) sb(i)],[est_cdf(i-1) est_cdf(i-1)],'Color','r','Linestyle','-','Linewidth',2);
  line([sb(i) sb(i)],[est_cdf(i-1) est_cdf(i)],'Color','r','Linestyle',':','Linewidth',1);
  if (strcmp(truemixing,'continuous'))
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(i-1))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(i))-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(i))-est_cdf(i))];
  else
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i-1),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i-1))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),0,tb,true_p)-est_cdf(i))];
    cdf_err=[cdf_err; abs(discrete_cdf(sb(i),1,tb,true_p)-est_cdf(i))];
  end
end
est_cdf_fh=line([sb(end) xu],[1 1],'Color','r','Linestyle','-','Linewidth',2);
if (strcmp(truemixing,'continuous'))
  cdf_err=[cdf_err; abs(mixed_normal_cdf(true_p,true_nparams,sb(end))-1)];
else
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),1,tb,true_p)-1)];
  cdf_err=[cdf_err; abs(discrete_cdf(sb(end),0,tb,true_p)-1)];
end
if (strcmp(truemixing,'continuous'))
legend(sprintf('Truth: mixture of %i normals',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Southeast');
else
legend([true_cdf_fh est_cdf_fh],sprintf('Truth: finite mixture with %i types',n_true_types),sprintf('Estimated: finite mixture with %i types',n_est_types),'Location','Southeast');
end
xlabel('\beta');
ylabel('Cumulative Distribution Function');
title({'True vs approximate CDF of random coefficients: minimum L^2 distance'},{sprintf('Maximum absolute error between true and approximate CDF: %g',max(cdf_err))});
axis('tight');
hold off;

end

if (ml_estimation);

  x=data(:,1);
  if (strcmp(truemixing,'continuous'));
    y=data(:,4);  % this uses the first of the 3 possible dependent variables in the 1000 x 4 matrix data
  else;
    if (n_true_types == 1);
       y=data(:,2);
    else;
       y=data(:,3);
    end;
  end;

  gendat=0;
  loadat=1;

  if (gendat);
    loadat=0;
  end;

  % generate new data below, by setting gendat=1. If you want to re-use the saved newly generated
  % data (for example to estimate different specifications for random coefficients on the same 
  % data set) then after running this with gendat=1 on subsequent runs set gendat=0 and loadat=1

  if (gendat);  % note: the true model is determined from the truemixing and n_true_types variables
                % that were set above
  
    n=10000;
    y=zeros(n,1);
    x=randn(n,1);

    if (strcmp(truemixing,'continuous'));
      thetatrue=randn(n_true_types*3,1);
      cp=cprob(x,thetatrue,T);
      fprintf('generating new (y,x) data for continuous, mixed normal specification with %i mixture components\n',n_true_types);
      fprintf('randomly chosen theta coefficients:\n');
      thetatrue
    else;
      thetatrue=randn(n_true_types*2,1);
      cp=cprob(x,thetatrue,T);
      fprintf('generating new (y,x) data for discrete mixture specification of random coefficients with %i mixture components\n',n_true_types);
      fprintf('randomly chosen theta coefficients:\n');
      thetatrue
    end;
    for i=1:n;
       u=rand(1,1);
       if (u <= cp(i));
         y(i)=1;
       end;
    end;
    save('x','x');
    save('y','y');
    save('thetatrue','thetatrue'); 
    save('truemixing','truemixing');
    save('n_true_types','n_true_types');

  else;

   if (loadat);
    load('y');
    load('x');
    load('thetatrue');
    load('truemixing');
    load('n_true_types');
   end;

  end;
 
  if (gendat == 0 & loadat == 0);
 
    if (strcmp(truemixing,'discrete'));
     if (n_true_types == 2);
       thetatrue=[-2 .1 .8 log(1/.2-1)]';
     else;
       thetatrue=[-2 .5]';
     end;
    else;
     % y is the 4th column of the dat matrix above
     thetatrue=[-2 .5 log(.2)]';
    end;

  end;

  if (strcmp(estmixing,'continuous'));
    theta=randn(3*n_est_types,1);
    lf=@(theta) nllf(y,x,theta,@cprob);
  else;
    theta=randn(2*n_est_types,1);
    lf=@(theta) nllf(y,x,theta,@bprob);
end;

options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'MaxFunEvals',6000);
[thetahat,lfmax,]=fminunc(lf,theta,options);
lfmax=-lfmax;

  % compute standard errors as inverse of information matrix

  if (strcmp(estmixing,'continuous'));
    [lfv,dllfv]=cprob(x,thetahat,T);
    im=inv(dllfv'*dllfv);
    stderr=sqrt(diag(im));
  else;
    [lfv,dllfv]=bprob(x,thetahat,T);
    im=inv(dllfv'*dllfv);
    stderr=sqrt(diag(im));
  end;

n=size(x,1);

if (strcmp(estmixing,'continuous'));
fprintf('Optimized value of the log-likelihood for %s normal(mu,sigma^2) specification of random coefficients: %g\n',estmixing,lfmax);
else;
fprintf('Optimized value of the log-likelihood for %s mixture specification of random coefficients with %i discrete types: %g\n',estmixing,n_est_types,lfmax);
end;
fprintf('Number of observations: %i\n',n);
fprintf('AIC: %g  BIC: %g\n',2*lfmax-2*size(thetahat,1),2*lfmax-log(n)*size(thetahat,1));
fprintf('\nEstimated parameter vector and standard errors\n');
[thetahat stderr]

if (strcmp(estmixing,'continuous'));
   fprintf('Estimated model with N(mu,sigma^2) random beta coefficient with %i mixture components\n',n_true_types);
end;

for i=1:n_est_types; 
 if (strcmp(estmixing,'continuous'));
  fprintf('mu(%i)=%g (%g)\n',i,thetahat(2+2*(i-1)),stderr(2+2*(i-1)));
  fprintf('sig(%i)=%g (%g)\n',i,exp(thetahat(3+2*(i-1))),exp(thetahat(3+2*(i-1)))*stderr(3+2*(i-1)));
 else;
  fprintf('beta(%i)=%g (%g)\n',i,thetatrue(1+i),stderr(1+i));
 end;
end;
if (n_est_types > 1);
 pt=probtype(n_est_types,thetahat,estmixing);
 for i=1:n_est_types;
    fprintf('Estimated probability of type %i: %g\n',i,pt(i));
 end;
end;

fprintf('\nTrue parameter vector\n');
thetatrue
if (strcmp(truemixing,'continuous'));
   fprintf('True model is a mixture of normals with %i mixture components\n',n_true_types);
else;
   fprintf('True model has a discrete distribution over beta with %i types\n',n_true_types);
end;

for i=1:n_true_types;
 if (strcmp(truemixing,'continuous'));
  fprintf('mu(%i)=%g\n',i,thetatrue(2+2*(i-1)));
  fprintf('sig(%i)=%g\n',i,exp(thetatrue(3+2*(i-1))));
 else;
  fprintf('beta(%i)=%g\n',i,thetatrue(1+i));
 end;
end;
if (n_true_types > 1);
  pt=probtype(n_true_types,thetatrue,truemixing);
  for i=1:n_true_types;
     fprintf('True probability of type %i: %g\n',i,pt(i));
  end;
end;

x=sort(x);

if (strcmp(estmixing,'continuous'));
  estprob=cprob(x,thetahat,T);
else;
  estprob=bprob(x,thetahat,T);
end;

if (strcmp(truemixing,'continuous'));
  trueprob=cprob(x,thetatrue,T);
else;
  trueprob=bprob(x,thetatrue);
end;



clf;
figure(1);
hold on;
plot(x,estprob,'r-','Linewidth',2);
plot(x,trueprob,'b-','Linewidth',2);
xlabel('x');
ylabel('Estimated CCP P(x)');
legend('Estimated CCP P(x)','True CCP P(x)','Location','Best');
if (strcmp(estmixing,'continuous'));
title({'Maximum likelihood estimate of P(x) using N(\mu,\sigma^2) distribution',sprintf('max difference between true and estimated P(x): %g',max(abs(trueprob-estprob)))});
else;
title({sprintf('Maximum likelihood estimate of P(x) using discrete mixture with %i types',n_est_types),sprintf('max difference between true and estimated P(x): %g',max(abs(trueprob-estprob)))});
end;
hold off;

end;
