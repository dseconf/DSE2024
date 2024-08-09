% mixed_logit_generator  creates data from mixed logit binary choice model
% John Rust, Georgetown University, October 2019

global nqp qw qa;

% generate quadrature weights, abscissae for normally distributed random coefficient on x
% in the binary logit model.

nqp=20;
[qa,qw]=quadpoints(nqp,0,1);
qa=icdf('normal',qa,0,1);


gendat=1;

alpha=-2;

b_no_rc=.5;


if (gendat);

n=1000000;

x=randn(n,1);
x=3+2*x;

choice0=zeros(n,1);
choice1=zeros(n,1);
choice2=zeros(n,1);

for i=1:n;

   u=rand(1,1);
   u0=rand(1,1);
   if (u0 < .2);
   prob0=1/(1+exp(alpha+b_no_rc*x(i)));
   choice0(i)=(u <= prob0);
   elseif (u0 >= .2 & u0 <.6);
   rbc=b_no_rc+.5+.2*randn(1,1);
   prob1=1/(1+exp(alpha+rbc*x(i)));
   choice0(i)=(u <= prob1);
   else;
   rbc=0.3*randn(1,1);
   prob2=1/(1+exp(alpha+rbc*x(i)));
   choice0(i)=(u <= prob2);
   end;

%   u=rand(1,1);
%   if (u <= .2);
%     rbc=.1;
%   else;
%     rbc=.8;
%   end;
%
%   u2=rand(1,1);
%   prob2=1/(1+exp(alpha+rbc*x(i)));
%   choice2(i)=(u2 <= prob2);

end;


diary('choice_data.dat');
%[x choice0 choice1 choice2]
[x choice0]
diary('off');

end;

plot_mixing=0;

if (plot_mixing);
truemixing='continuous';
estmixing='discrete';
n_est_types=5;
n_true_types=1;

x=(-1:.1:1)';
%x=(-.2:.01:0)';
%x=(-4:.1:4)';
n=size(x,1);
%trueprob=zeros(n,1);
%for i=1:n;
%    trueprob(i)=1/(1+exp(alpha+b_no_rc*x(i)));
%end;
thetatrue=[alpha .5 -.6 -1]';
thetatrue=[alpha b_no_rc]';

if (strcmp(truemixing,'continuous'));
  thetatrue=[alpha 2 1]';
  trueprob=cprob(x,thetatrue);
else;
  thetatrue=[alpha .5 -.6  1.1 -1 0]';
  trueprob=bprob(x,thetatrue);
end;

if (strcmp(estmixing,'continuous'));
  theta=randn(3,1);
  ssr=@(theta) sum((trueprob-cprob(x,theta)).^2);
else;
  if (n_est_types ==1);
  theta=randn(2,1);
  elseif (n_est_types == 2);
  theta=randn(4,1);
  elseif (n_est_types == 3);
  theta=randn(6,1);
  elseif (n_est_types == 4);
  theta=randn(8,1);
  elseif (n_est_types == 5);
  theta=randn(10,1);
  end;

  ssr=@(theta) sum((trueprob-bprob(x,theta)).^2);
end;

[thetahat,ssrmin]=fminunc(ssr,theta);

% use the solution to 3 type model as starting point for a 4 type model
%theta=[thetahat(1:4)' thetahat(4) thetahat(5:6)' -4 -4]';
%ssr=@(theta) sum((trueprob-bprob(x,theta)).^2);
%fprintf('gh\n');
%[thetahat,ssrmin]=fminunc(ssr,theta);

%[thetahat,ssrmin]=fminunc(ssr,thetahat);
fprintf('estimated and true theta\n');
fprintf('estimated theta\n');
thetahat'
fprintf('true theta\n');
thetatrue'
if (size(thetatrue,1) ==4);
   fprintf('true p1=%g\n',1/(1+exp(thetatrue(4))));
end;
if (size(thetatrue,1) ==6);
   fprintf('true p1=%g\n',1/(1+exp(thetatrue(5))+exp(thetatrue(6))));
   fprintf('true p2=%g\n',exp(thetatrue(5))/(1+exp(thetatrue(5))+exp(thetatrue(6))));
end;
if (size(thetahat,1) == 6);
   fprintf('estimated p1=%g\n',1/(1+exp(thetahat(5))+exp(thetahat(6))));
   fprintf('estimated p2=%g\n',exp(thetahat(5))/(1+exp(thetahat(5))+exp(thetahat(6))));
end;
fprintf('minimum objective function %g\n',ssrmin);

if (strcmp(estmixing,'continuous'));
  estprob=cprob(x,thetahat);
else;
  estprob=bprob(x,thetahat);
end;

clf;
figure(1);
hold on;
plot(x,trueprob,'b-','Linewidth',2);
plot(x,estprob,'r-','Linewidth',2);
if (strcmp(estmixing,'discrete'));
legend(sprintf('True, %s heterogeneity',truemixing),sprintf('Estimated, %s heterogeneity, %i types',estmixing,n_est_types),'Location','Best');
else;
legend(sprintf('True, %s heterogeneity',truemixing),sprintf('Estimated, %s heterogeneity',estmixing),'Location','Best');
end;
xlabel('x');
ylabel('P(1|x)');
title('True vs nonlinear least squares approximation to P(1|x)');
hold off;

end;
