% check_nllf2.m checks the derivatives produced by kldist.m in the panel/binomial case, T > 1
%                John Rust, Georgetown University, July 2024

%T=15;
%nobs=100;
%[trueprob,dtrueprob,true_p]=cprob1(x,thetatrue,T);
%[ydata,xdata,tv]=gendata(nobs,T,thetatrue,true_p,truemixing);

[lf,dlf,hlf]=nllf2(ydata,xdata,theta);

delt=1e-6;
nt=numel(theta);
ndlf=zeros(nt,1);
nhlf=zeros(nt,nt);

for i=1:nt

    thetau=theta; 
    thetau(i)=thetau(i)+delt;
    [lfu,dlfu]=nllf2(ydata,xdata,thetau);
    thetal=theta; 
    thetal(i)=thetal(i)-delt;
    [lfl,dlfl]=nllf2(ydata,xdata,thetal);

    ndlf(i)=(lfu-lfl)/(2*delt);
    nhlf(:,i)=(dlfu-dlfl)/(2*delt);

end

fprintf('absolute different between analytical and numerical gradient of the negative log-likeihood function\n');
abs(dlf-ndlf)
fprintf('absolute different between analytical and numerical hessian of the negative log-likeihood function\n');
abs(hlf-nhlf)
