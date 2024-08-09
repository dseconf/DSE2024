% nlls_deriv_checker.m: numerical check of analytical gradients of nlls function
% John Rust, Georgetown University, July 2024

[ssr,dssr]=nlls(trueprob,x,theta);
delta=1e-6;
dt=numel(theta);
ndssr=zeros(dt,1);

fprintf('Checking derivatives of nlls, with respect to its %i parameters\n',dt);

for i=1:dt
   e=zeros(dt,1);
   e(i)=1;
   ndssr(i)=(nlls(trueprob,x,theta+delta*e)-nlls(trueprob,x,theta-delta*e))/(2*delta);
   fprintf('%i analytical deriv: %g  numerical %g difference %g\n',i,dssr(i),ndssr(i),abs(dssr(i)-ndssr(i)));
end
