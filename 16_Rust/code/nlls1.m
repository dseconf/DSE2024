% nlls1.m nonlinear least squares criterion for the CCP
%         This is similar to nlls.m except that it calls the functions cprob1, bprob1, respectively instead of cprob,bprob
%         John Rust, Georgetown University, July 2024


function varargout=nlls1(trueprob,x,theta,T);

  if (nargout == 1)
      estprob=bprob1(x,theta,T);
  else
      [estprob,destprob,hestprob]=bprob1(x,theta,T);
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

      varargout{2}=-2*sum((trueprob-estprob).*destprob)';

      if (nargout > 2)
 
         if (T > 1)

           tmp1=zeros(sx,k,k);
           tmp2=zeros(k,k);
           for t=0:T
             tmp=(sqrt(trueprob(:,t+1))./estprob(:,t+1)).*squeeze(destprob(:,t+1,:));
             tmp2=tmp2+tmp'*tmp;
             tmp1=tmp1+(trueprob(:,t+1)./estprob(:,t+1)).*squeeze(hestprob(:,t+1,:,:));
           end

           varargout{3}=tmp2-squeeze(sum(tmp1));

         else

           varargout{3}=destprob'*destprob-squeeze(sum((trueprob-estprob).*hestprob));
           varargout{3}=2*varargout{3};

         end

      end

    end
 
  end

end
