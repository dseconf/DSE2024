% nllf1.m: (negative of) the mixed logit log likelihood function
%          John Rust, Georgetown University, July, 2024
%          (note, this is for the case of cross sectional data only,
%          use nllf2.m for the case of panel data, which includes the
%          panel data case as a special case, so nlff2.m subsumes this
%          function, but the two functions use different approaches to
%          calculate the log-likelihood even in the cross section case. 
%          This function calls bprof1.m to calculate the mixture of CCPs
%          and is fully "vectorized" to make it faster. But the drawback
%          is that it computes the gradient and hessian for all observations
%          in a single calls to bprob1 and hence for cases with large numbers
%          of observations and many parameters in theta, the array hmcp storing
%          the hessian of the log-likelihood observation by observation can be
%          huge and lead to memory problems. nllf2.m, in contrast, calls bprob2.m
%          in a do-loop that calculates the likelihood, its gradient and hessian
%          observation by observation, and thus is much less memory intensive though
%          potentially slower since it is not fully vectorized. Thus nllf2.m is
%          preferred for memory reasons for estimating models with large numbers of
%          unobserved types (i.e. networks with large numbers of hidden units)

  function varargout=nllf1(y,x,theta);

    [mcp,dmcp,hmcp]=bprob1(x,theta,1);

    k=numel(theta);

    varargout{1}=-sum(y.*log(mcp)+(1-y).*log(1-mcp));

    if (nargout > 1)

      wy=y./mcp-(1-y)./(1-mcp);

     varargout{2}=-(wy'*dmcp)';

     if (nargout > 2)

       tmp=wy.*dmcp;
       varargout{4}=tmp'*tmp;

       tmp=sqrt(y./(mcp.^2)+(1-y)./((1-mcp).^2));
       tmp=tmp.*dmcp;

       tmp1=(y./mcp-(1-y)./(1-mcp)).*dmcp;
      
       tmp2=(y./mcp-(1-y)./(1-mcp))'*hmcp(:,:);
       tmp2=reshape(tmp2,k,k); 
       varargout{3}=-tmp2+tmp'*tmp;

     end

    end


