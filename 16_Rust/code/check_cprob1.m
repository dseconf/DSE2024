% check_cprob1.m checks the derivatives produced by cprob1.m in the binomial case, T > 1
%               John Rust, Georgetown University, July 2024

T=1;
thetatrue=randn(11,1);
[bprobc,dbprob]=cprob1(x,thetatrue,T);

delt=1e-6;
ndprob=zeros(size(dbprob));
nt=numel(thetatrue);

for i=1:nt

    thetau=thetatrue; 
    thetau(i)=thetau(i)+delt;
    bprobu=cprob1(x,thetau,T);
    thetal=thetatrue; 
    thetal(i)=thetal(i)-delt;
    bprobl=cprob1(x,thetal,T);

    if (T > 1)
      ndbprob(:,:,i)=(bprobu-bprobl)/(2*delt);
    else
      ndbprob(:,i)=(bprobu-bprobl)/(2*delt);
    end

end

abs(dbprob-ndbprob)
