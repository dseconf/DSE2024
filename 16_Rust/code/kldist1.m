% kldist1.m: computes the Kullback-Leibler distance between a true probability distribution and an approximate (parametric) one
%            This is similar to kldist.m except that it calls the functions cprob1, bprob1, respectively instead of cprob,bprob
%            reflecting the case where there is a random intercept as well as slope coefficient in the model
%
%            John Rust, Georgetown University, July, 2024

  function varargout=kldist1(trueprob,x,type,theta,T);

   k=numel(theta);
   sx=size(x,1);
   hestprob=0;

   if (strcmp(type,'cprob'))
     if (nargout == 1)
      estprob=cprob1(x,theta);
     else
      [estprob,destprob]=cprob1(x,theta,T);
     end
   else
     if (nargout == 1)
      estprob=bprob1(x,theta,T);
     else
      if (nargout == 2)
      [estprob,destprob]=bprob1(x,theta,T);
      else
      [estprob,destprob,hestprob]=bprob1(x,theta,T);
      end
     end
   end

   if (T > 1)

      varargout{1}=-sum(sum((trueprob.*log(estprob./trueprob))'));
  
   else

      varargout{1}=-sum(trueprob.*log(estprob./trueprob)+(1-trueprob).*log((1-estprob)./(1-trueprob)));

   end

   if (nargout > 1)

     if (T > 1)

        tmp1=zeros(k,1);
        for i=1:k
           tmp=squeeze(destprob(:,:,i));
           tmp1(i)=-sum(sum((trueprob./estprob).*tmp)');
        end
        varargout{2}=tmp1;

     else

       varargout{2}=-sum((trueprob./estprob-(1-trueprob)./(1-estprob)).*destprob)';

     end

   end

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

       tmp=sqrt(trueprob./(estprob.^2)+(1-trueprob)./((1-estprob).^2)).*destprob;
       varargout{3}=tmp'*tmp-squeeze(sum((trueprob./estprob-(1-trueprob)./(1-estprob)).*hestprob));

     end

   end

  end
     
