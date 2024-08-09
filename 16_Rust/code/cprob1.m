% cprob1.m: function to compute the choice probability as a function of x for a 
%            normal continuous mixing distribution for the random coefficient beta in 
%            a binary logit model where all coefficients including the intercept are 
%            random (i.e. heterogeneous). Thus p(1|x,alpha,beta)=1/(1+exp(x*beta), where
%            beta is a 2x1 vector corresponding to a type-specific intercept and slope
%            where x*beta=beta(1)+beta(2)*x where x is a scalar covariate. In this case
%            the parameters theta are the parameters of a mixture of normal specification
%            so each "type" is defined by the 5 parameters if a bivariate normal distributiong 
%            defining the (beta(1),beta(2)) joint distribution for that type. Thus if there are k types
%            there are 5*k parameters defining the k bivariate normal distributions entering the 
%            mixture and the last k-1 elements of theta are parameters defining a multinomial
%            logit mixture of the k normal distributions. Thus theta will have 5*k+k-1 parameters
%            in total where the last k-1 parameters determine this mixing distribution.
%            This program is similar to cprob.m except that it makes the constant term to be
%            heterogeneous whereas in cprob.m the constant term is assumed the same for all individuals
%            and only the slope coefficient on x is heterogeneous
%
%            The 5 parameters for the bivariate normal are determined as follows
%            beta(1) is univariate normal with a  mean of theta(1)
%            exp(theta(2)) is the variance of beta(1)
%            beta(2)=theta(3)+theta(5)*beta(1)+epsilon(2) where epsilon(2) is univariate normal
%            with mean 0 and var(epsilon(2))=exp(theta(4)).
%            Thus if theta(5) is non-zero then beta(1) and beta(2) are correlated with covariance equal to
%            cov(beta(1),beta(2)=theta(5)*exp(theta(2))
%
%            John Rust Georgetown University, July 2024

function varargout=cprob1(x,theta,T);   % choice probability, gradient after integrating
                                        % over mixture of continuous bivariate normal distribution for random 
                                        % beta coefficients, whered dcpv is the derivative
                                        % of the ccp with respect to the parameters

   global qa qw nqp;  % nqp: number quadrature points, qw are quadrature weights
                      %      qa quadrature abcissae, transformed by inverse of 
                      %      a normal CDF

    sx=size(x,1);
    k=numel(theta);
    nt=(k+1)/6; % number of mixtures in the mixture of normals

    if (nt == 1)  % only 1 mixture component

      mu0=theta(1);
      sig0=exp(theta(2)/2);
      mu1=theta(3);
      sig1=exp(theta(4)/2); 
      b=theta(5);

      cpv=zeros(sx,1);

      if (nargout > 1)
        dcpv=zeros(sx,5);
      end

      for i=1:nqp
        beta0=mu0+sig0*qa(i);
        cpvj=zeros(sx,1);
        dcpvj=zeros(sx,5);
        for j=1:nqp
          p=1./(1+exp(beta0+x*(mu1+b*beta0+sig1*qa(j))));
          cpvj=cpvj+qw(j)*p;
          p1mp=p.*(1-p);
          if (nargout > 1)
              dcpvj(:,1)=dcpvj(:,1)-qw(j)*p1mp.*(1+x*b);
              dcpvj(:,2)=dcpvj(:,2)-qw(j)*p1mp.*(1+x*b)*sig0*qa(i)/2;
              dcpvj(:,3)=dcpvj(:,3)-qw(j)*p1mp.*x;
              dcpvj(:,4)=dcpvj(:,4)-qw(j)*p1mp.*x*sig1*qa(j)/2;
              dcpvj(:,5)=dcpvj(:,5)-qw(j)*p1mp.*x*beta0;
          end
        end
        cpv=cpv+qw(i)*cpvj;
        if (nargout > 1)
          dcpv=dcpv+qw(i)*dcpvj;
        end
      end

      varargout{1}=cpv;

      if (nargout > 1)
        varargout{2}=dcpv;
        varargout{3}=1;
        varargout{4}=[mu0; sig0; b*mu0+mu1; sqrt((b^2)*(sig0^2)+(sig1^2)); b];
                     % provide the marginal mean and std deviation of the mixture components and info to compute their covariances
                     % first two columns store mean, std for the normal distribution of intercept
                     % columns 3, 4 store mean, std for the normal distribution of the slope
                     % column 5 stores b coefficient relating beta(2) (coefficient of x) to beta(1) (intercept) 
      end

      if (T > 1)  % panel data

        % in this case the first output argument is a binomial distribution of possible 0/1 choices
        % evaluated at a vector of probabilities corresponding to a vector x of covariates. Thus, the
        % return in this case is a matrix with dim(x) rows and T+1 columns where in the binomial the possible outcomes range
        % for t=0,1,...,T
        % Therefore the gradient of the choice probability is a 3-dimensional array where 1st dimension is x,
        % second dimension is t (from 0 to T) and 3rd dimension is the number of parameters, which is 3 in this case

        binprobs=zeros(sx,T+1);
        dbinprobs=zeros(sx,T+1,5);
        for t=0:T
          binprobs(:,t+1)=binopdf(t,T,cpv);
          if (nargout > 1)
          dbinprobs(:,t+1,:)=binprobs(:,t+1).*(t./cpv-(T-t)./(1-cpv)).*dcpv;
          end
        end

        varargout{1}=binprobs;
        if (nargout > 1)
          varargout{2}=dbinprobs;
        end

      end

    else  % nt > 1 mixture components

      mu0=zeros(nt,1);
      sig0=zeros(nt,1);
      mu1=zeros(nt,1);
      sig1=zeros(nt,1);
      b=zeros(nt,1);
      p=zeros(nt,1);
      cpv=zeros(sx,nt);
      mcp=zeros(sx,1);
      p=zeros(nt,1);

      if (nargout > 1)

       dpdt=zeros(nt,nt-1);
       dmcpdt=zeros(sx,k);

      end

      for t=1:nt
        if (t ==1)
           p(t)=1;
        else
           p(t)=exp(theta(5*nt+t-1));
        end
      end
      p=p/sum(p);
   
      for t=1:nt-1
        dpdt(1,t)=-p(1)*p(t+1);
      end

      for i=2:nt
         for j=1:nt-1
            if (j==i-1)
              dpdt(i,j)=p(i)*(1-p(i));
            else
              dpdt(i,j)=-p(i)*p(j+1); 
            end
         end
      end

      for t=1:nt

        mu0(t)=theta(1+(t-1)*5);
        sig0(t)=exp(theta(2+(t-1)*5)/2);
        mu1(t)=theta(3+(t-1)*5);
        sig1(t)=exp(theta(4+(t-1)*5)/2);
        b(t)=theta(5+(t-1)*5);

        dcpv=zeros(sx,5);
        cpv=zeros(sx,1);

        for i=1:nqp

          beta0=mu0(t)+sig0(t)*qa(i);
          cpvj=zeros(sx,1);
          dcpvj=zeros(sx,5);

          for j=1:nqp

            pv=1./(1+exp(beta0+x*(mu1(t)+b(t)*beta0+sig1(t)*qa(j))));
            cpvj=cpvj+qw(j)*pv;
            p1mp=pv.*(1-pv);
            if (nargout > 1)
              dcpvj(:,1)=dcpvj(:,1)-qw(j)*p1mp.*(1+x*b(t));
              dcpvj(:,2)=dcpvj(:,2)-qw(j)*p1mp.*(1+x*b(t))*sig0(t)*qa(i)/2;
              dcpvj(:,3)=dcpvj(:,3)-qw(j)*p1mp.*x;
              dcpvj(:,4)=dcpvj(:,4)-qw(j)*p1mp.*x*sig1(t)*qa(j)/2;
              dcpvj(:,5)=dcpvj(:,5)-qw(j)*p1mp.*x*beta0;
            end
          end

          cpv=cpv+qw(i)*cpvj;

          if (nargout > 1)
             dcpv=dcpv+qw(i)*dcpvj;
          end

        end

        mcp=mcp+p(t)*cpv;

        if (nargout > 1)

          dmcpdt(:,1+5*(t-1):5*t)=p(t)*dcpv;
          for j=1:nt-1
            dmcpdt(:,nt*5+j)=dmcpdt(:,nt*5+j)+dpdt(t,j)*cpv;
          end

        end

      end  % end of main loop over types
 
      if (nargout > 1)
        varargout{2}=dmcpdt;
        varargout{3}=p;
        mn_params=zeros(5,nt);
        for t=1:nt
           mn_params(1,t)=mu0(t);
           mn_params(2,t)=sig0(t);
           mn_params(3,t)=b(t)*mu0(t)+mu1(t);
           mn_params(4,t)=sqrt((b(t)^2)*(sig0(t)^2)+sig1(t)^2);
           mn_params(5,t)=b(t);
        end
        varargout{4}=mn_params;

                     % provide the 5 parameters defining the mixture components of the bivariate mixed normal distribution
                     % first two columns store mean, std for the normal distribution of intercept
                     % columns 3, 4 store mean, std for the normal distribution of the slope
                     % column 5 stores b coefficient relating beta(2) (coefficient of x) to beta(1) (intercept) 
      end

      varargout{1}=mcp;

      if (T > 1)  % panel data

        % in this case the first output argument is a binomial distribution of possible 0/1 choices
        % evaluated at a vector of probabilities corresponding to a vector x of covariates. Thus, the
        % return in this case is a matrix with dim(x) rows and T+1 columns where in the binomial the possible outcomes range
        % for t=0,1,...,T
        % Therefore the gradient of the choice probability is a 3-dimensional array whose 1st dimension is x,
        % second dimension is t (from 0 to T) and 3rd dimension is the number of parameters, which is dimension of theta

        binprobs=zeros(sx,T+1);
        dbinprobs=zeros(sx,T+1,k);
        for t=0:T
          binprobs(:,t+1)=binopdf(t,T,mcp);
          if (nargout > 1)
          dbinprobs(:,t+1,:)=binprobs(:,t+1).*(t./mcp-(T-t)./(1-mcp)).*dmcpdt;
          end
        end

        varargout{1}=binprobs;
        if (nargout > 1)
          varargout{2}=dbinprobs;
        end

      end

    end
