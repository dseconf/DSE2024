% nlls.m nonlinear least squares criterion for the CCP
%        John Rust, Georgetown University, July 2024


  function varargout=nlls(trueprob,x,theta,T);

  if (nargout == 1)
      estprob=bprob(x,theta,T);
  else
      [estprob,destprob]=bprob(x,theta,T);
  end

  varargout{1}=sum(sum((trueprob-estprob).^2)');

  if (nargout > 1)

    if (T > 1)

      sx=size(x,1);
      tmp=zeros(sx,1);
      for t=0:T
         tmp=tmp+(trueprob(:,t+1)-estprob(:,t+1)).*squeeze(destprob(:,t+1,:));
      end
      varargout{2}=-2*sum(tmp)';
    
    else

    varargout{2}=-2*sum((trueprob-estprob).*destprob);

    end
 
  end

  end
