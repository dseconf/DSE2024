% klhess.m: computes tha hessian matrix of the Kullback-Leibler distance between a true probability distribution and an approximate (parametric) one
%           John Rust, Georgetown University, July, 2024

  function [klhessv]=klhess(trueprob,x,type,theta);

   delt=1e-6;

   dt=numel(theta);
  
   klhessv=zeros(dt,dt); 

   for i=1:dt;
    
      thetau=theta;
      thetau(i)=thetau(i)+delt;
      [kldu,gkldu]=kldist(trueprob,x,type,thetau);

      thetal=theta;
      thetal(i)=thetal(i)-delt;
      [kldl,gkldl]=kldist(trueprob,x,type,thetal);
      
      klhessv(:,i)=(gkldu-gkldl)/(2*delt); 

   end

  end
     
