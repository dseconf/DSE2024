% check_bprob11.m checks the derivatives produced by bprob1.m in the binomial case, T > 1
%                John Rust, Georgetown University, July 2024

T=1;
theta=randn(11,1);
[bprobc,dbprob,hbprob]=bprob1(x,theta,T);
delt=1e-6;
ndbprob=zeros(size(dbprob));
nhbprob=zeros(size(hbprob));
nt=numel(theta);

for i=1:nt

    thetau=theta; 
    thetau(i)=thetau(i)+delt;
    [bprobu,dbprobu,hbprobu]=bprob1(x,thetau,T);
    thetal=theta; 
    thetal(i)=thetal(i)-delt;
    [bprobl,dbprobl,hbprobl]=bprob1(x,thetal,T);

    if (T == 1)
      ndbprob(:,i)=(bprobu-bprobl)/(2*delt);
    else
      ndbprob(:,:,i)=(bprobu-bprobl)/(2*delt);
    end

    if (T == 1)
      nhbprob(:,:,i)=(dbprobu-dbprobl)/(2*delt);
    else
      nhbprob(:,:,:,i)=(dbprobu-dbprobl)/(2*delt);
    end

end

fprintf('Absolute difference between numerical and analytical gradients of CCP with respect to the %i parameters in theta\n',nt);
abs(dbprob-ndbprob)

fprintf('Absolute difference between numerical and analytical hessian of CCP with respect to the %i parameters in theta\n',nt);
abs(hbprob-nhbprob)
