% check_bprob2.m checks the derivatives produced by bprob1.m in the binomial case, T > 1
%                John Rust, Georgetown University, July 2024

check_type_probs=0;  % set to 1 to check the gradient and hessian of the multinomial logit type
                     % probabilities for the ntypes different types in the model with respect to  
                     % the ntypes-1 parameters defining them (the last ntypes-1 components of the 
                     % vector theta)

[bprobc,dbprob,hbprob,typeprob,typeparams,dtypeprob,htypeprob]=bprob2(y,x,theta);
delt=1e-6;
ndlogprob=zeros(size(dbprob));
nhlogprob=zeros(size(hbprob));
nhbprob=zeros(size(hbprob));
nt=numel(theta);

dlogprob=dbprob/bprobc;
hlogprob=hbprob/bprobc-(dbprob*dbprob')/(bprobc^2);

ntypes=(nt+1)/3;

nhtypeprob=zeros(ntypes-1,ntypes-1,ntypes);
ndtypeprob=zeros(ntypes,nt);

for i=1:nt

    thetau=theta; 
    thetau(i)=thetau(i)+delt;
    [bprobu,dbprobu,hbprobu,typeprobu,typeparamsu,dtypeprobu,htypeprobu]=bprob2(y,x,thetau);
    thetal=theta; 
    thetal(i)=thetal(i)-delt;
    [bprobl,dbprobl,hprobl,typeprobl,typeparamsl,dtypeprobl,htypeprobl]=bprob2(y,x,thetal);

    ndbprob(i)=(bprobu-bprobl)/(2*delt);
    ndlogprob(i)=(log(bprobu)-log(bprobl))/(2*delt);
    nhlogprob(:,i)=(dbprobu/bprobu-dbprobl/bprobl)/(2*delt);
    nhbprob(:,i)=(dbprobu-dbprobl)/(2*delt);

    if (check_type_probs)

      ndtypeprob(:,i)=(typeprobu-typeprobl)/(2*delt);

      if (i > 2*ntypes)

        for j=1:ntypes

          nhtypeprob(:,i-2*ntypes,j)=((dtypeprobu(j,:)-dtypeprobl(j,:))')/(2*delt);

        end

      end

    end

end


fprintf('Checking gradient of log of mixed f(y|x)  with respect to all %i parameters of the mixture model\n',nt);
abs(dlogprob-ndlogprob)
fprintf('Checking hessian of log of mixed f(y|x)  with respect to all %i parameters of the mixture model\n',nt);
abs(hlogprob-nhlogprob)


if (check_type_probs)

fprintf('Checking gradient of type probabilities with respect to type probability parameters\n');
fprintf('numerical\n');
ndtypeprob(:,2*ntypes+1:end)
fprintf('analytical\n');
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
