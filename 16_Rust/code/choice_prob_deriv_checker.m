% choice_prob_deriv_checker.m: numerical check of analytical gradients of cprob and bprob functions
% John Rust, Georgetown University, July 2024

% first part is to check cprob.m function, the mixture of normals specification for random coefficients

x=randn(2,1);
delta=1e-6;
theta=randn(6,1);
sx=size(x,1);
if (sx == 1)
  if (size(x,2) > 1)
     x=x';
     sx=numel(x);
  end
end

dt=size(theta,1);

[cp,dcp]=cprob(x,theta);

ndcp=zeros(sx,dt);

fprintf('Checking derivatives of cprob, mixture of normals specification with %i mixture components\n',dt/3);

for i=1:dt;
   e=zeros(dt,1);
   e(i)=1;
   ndcp(:,i)=(cprob(x,theta+delta*e)-cprob(x,theta-delta*e))/(2*delta);
   if (sx ==1);
   fprintf('%i analytical deriv: %g  numerical %g difference %g\n',i,dcp(i),ndcp(i),abs(dcp(i)-ndcp(i)));
   else;
   fprintf('deriv %i        analytical,          numerical,            absolute difference\n',i);
   [dcp(:,i) ndcp(:,i) abs(dcp(:,i)-ndcp(:,i))]
   end;
end;

% this part is to check bprob.m function, the finite mixture of point masses on beta random coefficients

x=randn(2,1);
delta=1e-6;
theta=randn(6,1);
sx=size(x,1);

dt=size(theta,1);

[cp,dcp]=bprob(x,theta);

ndcp=zeros(sx,dt);

fprintf('Checking derivatives of bprob, mixture of point masses with %i mixture components\n',dt/2);

for i=1:dt;
   e=zeros(dt,1);
   e(i)=1;
   ndcp(:,i)=(bprob(x,theta+delta*e)-bprob(x,theta-delta*e))/(2*delta);
   if (sx ==1);
   fprintf('%i analytical deriv: %g  numerical %g difference %g\n',i,dcp(i),ndcp(i),abs(dcp(i)-ndcp(i)));
   else;
   fprintf('deriv %i        analytical,          numerical,            absolute difference\n',i);
   [dcp(:,i) ndcp(:,i) abs(dcp(:,i)-ndcp(:,i))]
   end;
end;
