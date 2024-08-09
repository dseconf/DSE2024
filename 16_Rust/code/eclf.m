% eclf.m: (negative of) the log likelihood function for the estimation-classification
%         algorithm for unobserved types
%
%         John Rust, Georgetown University, July, 2024
%
% varargout{1} is the value of the negative of the log-likehood at (y,x,theta)
% varargout{2} is the gradient of the negative of the log-likehood at (y,x,theta)
% varargout{3} is the hessian of the negative of the log-likehood at (y,x,theta)
% varargout{4} is the information matrix of the log-likehood at (y,x,theta)
% varargout{5} is the the vector of type classifications for each subject

  function varargout=eclf(y,x,theta);

    k=numel(theta);
    T=size(y,2);
    n=size(y,1);

    ntypes=k/2;

    llf=0;
    dllf=zeros(k,1);
    hllf=zeros(k,k);
    im=zeros(k,k);

    typevec=zeros(n,1);

    if (nargout > 3)
       im=zeros(k,k);
    end

    for i=1:n

      llfimax=0;
      dllfimax=zeros(2,1,ntypes);
      hllfimax=zeros(2,2,ntypes);
      maxtype=0;

      for j=1:ntypes

        [llfi,dllfi,hllfi]=bprob1(y(i,:)',x(i,:)',theta);

        dllfimax(:,j)=dllfi;
        hllfimax(:,:,j)=hllfi;

        if (maxtype == 0)
           maxtype=1;
           llfimax=llfi;
        else
           if (llfi > llfimax)
             maxtype=j;
             llfimax=llfi;
           end
        end

      end

      typevec(i)=maxtype;

      llf=llf-log(llfimax);

      dlogprob=dllfimax(:,maxtype)/llfimax;
      dllf=dllf-dlogprob;
  
      hlogprob=hllfimax(:,:,maxtype)/llfimax-(dllfimax(:,maxtype)*dllfi(:,maxtype)')/(llfimax^2);

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
       if (nargout > 4)
         varargout{5}=typevec;
       end
    end

  end


