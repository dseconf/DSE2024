% check_bprob1.m checks the derivatives produced by bprob1.m in the binomial case, T > 1
%                John Rust, Georgetown University, July 2024

check_type_probs=0;  % set to 1 to check the gradient and hessian of the multinomial logit type
                     % probabilities for the ntypes different types in the model with respect to  
                     % the ntypes-1 parameters defining them (the last ntypes-1 components of the 
                     % vector theta)

T=1;
theta=randn(2,1);
[bprobc,dbprob,hbprob,typeprob,typeparams,dtypeprob,htypeprob]=bprob1(x,theta,T);
delt=1e-6;
ndbprob=zeros(size(dbprob));
nhbprob=zeros(size(hbprob));
nt=numel(theta);

ntypes=(nt+1)/3;

nhtypeprob=zeros(ntypes-1,ntypes-1,ntypes);
ndtypeprob=zeros(ntypes,nt);

for i=1:nt

    thetau=theta; 
    thetau(i)=thetau(i)+delt;
    [bprobu,dbprobu,hbprobu,typeprobu,typeparamsu,dtypeprobu,htypeprobu]=bprob1(x,thetau,T);
    thetal=theta; 
    thetal(i)=thetal(i)-delt;
    [bprobl,dbprobl,hbprobl,typeprobl,trueparamsl,dtypeprobl,htypeprobl]=bprob1(x,thetal,T);

    if (T == 1)
      ndbprob(:,i)=(bprobu-bprobl)/(2*delt);
      nhbprob(:,:,i)=(dbprobu-dbprobl)/(2*delt);
    else
      ndbprob(:,:,i)=(bprobu-bprobl)/(2*delt);
      for t=0:T
        nhbprob(:,t+1,:,i)=(dbprobu(:,t+1,:)-dbprobl(:,t+1,:))/(2*delt);
      end
    end

    if (check_type_probs)

      ndtypeprob(:,i)=(typeprobu-typeprobl)/(2*delt);

      if (i > 2*ntypes)

        for j=1:ntypes

          nhtypeprob(:,i-2*ntypes,j)=((dtypeprobu(j,:)-dtypeprobl(j,:))')/(2*delt);

        end

      end

    end

end

fprintf('Checking gradient of CCP with respect to all %i parameters of the mixture model\n',nt);
abs(dbprob-ndbprob)
fprintf('Checking hessian of CCP with respect to all %i parameters of the mixture model\n',nt);
abs(hbprob-nhbprob)

if (check_type_probs)

fprintf('Checking gradient of type probabilities with respect to type probability parameters\n');
fprintf('numerical\n');
ndtypeprob(:,2*ntypes+1:end)
fprintf('analyticcal\n');
dtypeprob
fprintf('Difference\n');
abs(dtypeprob-ndtypeprob(:,2*ntypes+1:end))

fprintf('Checking hessian of type probabilities with respect to type probability parameters\n');
fprintf('numerical\n');
nhtypeprob
fprintf('analytical\n');
htypeprob
fprintf('Difference\n');
abs(nhtypeprob-htypeprob)

end
