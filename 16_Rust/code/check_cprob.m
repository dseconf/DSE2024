% check_bprob.m checks the derivatives produced by bprob.m in the binomial case, T > 1
%               John Rust, Georgetown University, July 2024

T=4;
[bprobc,dbprob]=cprob(x,theta,T);

delt=1e-6;
ndprob=zeros(size(dbprob));
nt=numel(theta);

for i=1:nt

    thetau=theta; 
    thetau(i)=thetau(i)+delt;
    bprobu=cprob(x,thetau,T);
    thetal=theta; 
    thetal(i)=thetal(i)-delt;
    bprobl=cprob(x,thetal,T);

    if (T == 1)
      ndbprob(:,i)=(bprobu-bprobl)/(2*delt);
    else
      ndbprob(:,:,i)=(bprobu-bprobl)/(2*delt);
    end

end

abs(dbprob-ndbprob)
