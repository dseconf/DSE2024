% kldist_deriv_checker.m: numerical check of analytical gradients of kldist function
% John Rust, Georgetown University, July 2024



[kld,dkld]=kldist(trueprob,x,'bprob',theta);
delta=1e-6;
dt=numel(theta);
ndkld=zeros(dt,1);

fprintf('Checking derivatives of kldist, with respect to its %i parameters\n',dt);

for i=1:dt
   e=zeros(dt,1);
   e(i)=1;
   ndkld(i)=(kldist(trueprob,x,'bprob',theta+delta*e)-kldist(trueprob,x,'bprob',theta-delta*e))/(2*delta);
   fprintf('%i analytical deriv: %g  numerical %g difference %g\n',i,dkld(i),ndkld(i),abs(dkld(i)-ndkld(i)));
end
