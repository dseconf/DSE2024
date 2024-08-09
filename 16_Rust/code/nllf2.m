% nllf2.m: (negative of) the mixed logit log likelihood function
%          John Rust, Georgetown University, July, 2024
%
%          This function calculates the log-likelihood function via
%          repeated calls to bprob2.m inside a do-loop over all observations
%          and it gives correct returns for either cross-sectional or
%          panel data. Use nllf1.m in case of cross sectional data when
%          number of unobserved types is not too large since it is vectorized
%          and potentially faster than this function. But the drawback of nlls1.m
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

% varargout{1} is the value of the negative of the log-likehood at (y,x,theta)
% varargout{2} is the gradient of the negative of the log-likehood at (y,x,theta)
% varargout{3} is the hessian of the negative of the log-likehood at (y,x,theta)
% varargout{4} is the information matrix of the log-likehood at (y,x,theta)

  function varargout=nllf2(y,x,theta);

    k=numel(theta);
    T=size(y,2);
    n=size(y,1);

    llf=0;
    dllf=zeros(k,1);
    hllf=zeros(k,k);

    if (nargout > 3)
       im=zeros(k,k);
    end

    for i=1:n

      [llfi,dllfi,hllfi]=bprob2(y(i,:)',x(i,:)',theta);

      llf=llf-log(llfi);

      dlogprob=dllfi/llfi;
      dllf=dllf-dlogprob;
  
      hlogprob=hllfi/llfi-(dllfi*dllfi')/(llfi^2);

      hllf=hllf-hlogprob;

      if (nargout > 3)
         im=im+dlogprob*dlogprob';
      end

    end

    varargout{1}=llf;
    varargout{2}=dllf;
    varargout{3}=hllf;

    if (nargout > 3)
       varargout{4}=im;
    end

  end


