% check_nlls.m checks the derivatives produced by nlls.m in the panel/binomial case, T > 1
%              John Rust, Georgetown University, July 2024

T=1;
trueprob=cprob1(x,thetatrue,T);
[kld,dkld,hkld]=nlls1(trueprob,x,theta,T);

delt=1e-6;
nt=numel(theta);
ndkld=zeros(nt,1);
nhkld=zeros(nt,nt);

for i=1:nt

    thetau=theta; 
    thetau(i)=thetau(i)+delt;
    [kldu,dkldu]=nlls1(trueprob,x,thetau,T);
    thetal=theta; 
    thetal(i)=thetal(i)-delt;
    [kldl,dkldl]=nlls1(trueprob,x,thetal,T);

    ndkld(i)=(kldu-kldl)/(2*delt);
    nhkld(:,i)=(dkldu-dkldl)/(2*delt);

end

fprintf('absolute difference between analytical and numerical gradient of the L^2 distance\n');
abs(dkld-ndkld)
fprintf('absolute difference between analytical and numerical hessian of the L^2 distance\n');
abs(hkld-nhkld)
