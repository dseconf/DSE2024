% check_kldist.m checks the derivatives produced by kldist.m in the panel/binomial case, T > 1
%                John Rust, Georgetown University, July 2024

T=1;
trueprob=cprob1(x,thetatrue,T);
[kld,dkld,hkld]=kldist1(trueprob,x,'bprob',theta,T);

delt=1e-6;
nt=numel(theta);
ndkld=zeros(nt,1);
nhkld=zeros(nt,nt);

for i=1:nt

    thetau=theta; 
    thetau(i)=thetau(i)+delt;
    [kldu,dkldu]=kldist1(trueprob,x,'bprob',thetau,T);
    thetal=theta; 
    thetal(i)=thetal(i)-delt;
    [kldl,dkldl]=kldist1(trueprob,x,'bprob',thetal,T);

    ndkld(i)=(kldu-kldl)/(2*delt);
    nhkld(:,i)=(dkldu-dkldl)/(2*delt);

end

fprintf('absolute different between analytical and numerical gradient of the KL distance\n');
abs(dkld-ndkld)
fprintf('absolute different between analytical and numerical hessian of the KL distance\n');
abs(hkld-nhkld)
