% cprob.m: function to compute the choice probability as a function of x for a 
%          normal continuous mixing distribution for the random coefficient beta in 
%          a binary logit model, so theta(1) is the alpha parameter of the logit,
%          p(1|x,alpha,beta)=1/(1+exp(alpha+x*beta), and theta(2),theta(3) are the
%          (mu,sigma^2) parameters of the normal mixng distribution over beta so
%          we write  p(1|x)= E{P(1|theta(1)+x*beta)|theta(2),theta(3))} where
%          beta ~ N(theta(2),exp(theta)). If dimension of theta is larger than 3,
%          then the program computes a mixture of normals, where the theta(1) parameter
%          is the common intercept for all mixture components and there are separate
%          (mu,sigma^2) parameters for each mixture component with sigma=exp(theta(j))
%          where theta(j) is the component of theta for the normal standard deviation.
%          Thus, with k types there are a total of 1+k*2+k-1 parameters to be estimated.
%          John Rust Georgetown University, July 2024

function varargout=cprob(x,theta,T);   % choice probability, gradient after integrating
                                       % over continuous normal distribution for random 
                                       % beta coefficient, and dcpv is the derivative
                                       % with respect to the parameters

   global qa qw nqp;  % nqp: number quadrature points, qw are quadrature weights
                      %      qa quadrature abcissae, transformed by inverse of 
                      %      a normal CDF

    sx=size(x,1);
    k=size(theta,1);
    nt=k/3; % number of mixtures in the mixture of normals

    if (nt == 1)  % only 1 mixture component

      mu=theta(2);
      ss=exp(theta(3));

      p=1./(1+exp(theta(1)+x*(mu+ss*qa(1))));
     
      cpv=qw(1)*p;

      if (nargout > 1)
        dcpv=zeros(size(x,1),3);
        dcpv(:,1)=-qw(1)*p.*(1-p);
        dcpv(:,2)=-qw(1)*p.*(1-p).*x;
        dcpv(:,3)=-ss*qa(1)*qw(1)*p.*(1-p).*x;
      end

      for i=2:nqp
        p=1./(1+exp(theta(1)+x*(mu+ss*qa(i))));
        cpv=cpv+qw(i)*p;
        if (nargout > 1)
          dcpv(:,1)=dcpv(:,1)-qw(i)*p.*(1-p);
          dcpv(:,2)=dcpv(:,2)-qw(i)*p.*(1-p).*x;
          dcpv(:,3)=dcpv(:,3)-ss*qa(i)*qw(i)*p.*(1-p).*x;
          varargout{2}=dcpv;
          varargout{3}=1;
          varagrout{4}=[mu; ss];
        end
      end

      varargout{1}=cpv;

      if (T > 1)  % panel data

        % in this case the first output argument is a binomial distribution of possible 0/1 choices
        % evaluated at a vector of probabilities corresponding to a vector x of covariates. Thus, the
        % return in this case is a matrix with dim(x) rows and T+1 columns where in the binomial the possible outcomes range
        % for t=0,1,...,T
        % Therefore the gradient of the choice probability is a 3-dimensional array where 1st dimension is x,
        % second dimension is t (from 0 to T) and 3rd dimension is the number of parameters, which is 3 in this case

        binprobs=zeros(sx,T+1);
        dbinprobs=zeros(sx,T+1,3);
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

      mu=zeros(nt,1);
      sig=zeros(nt,1);
      p=zeros(nt,1);
      cpv=zeros(sx,nt);
      mcp=zeros(sx,1);
      if (nargout > 1)
       dpdt=zeros(nt,nt-1);
       dcdt=cell(nt,1);
       dmcpdt=zeros(sx,k);
      end

      for i=1:nt
        mu(i)=theta(2+(i-1)*2);
        sig(i)=exp(theta(3+(i-1)*2));
        if (i ==1)
           p(i)=1;
        else
           p(i)=exp(theta(1+2*nt+i-1));
        end
      end
      p=p/sum(p);
   
      for i=1:nt-1
        dpdt(1,i)=-p(1)*p(i+1);
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

      for i=1:nt
        dcdt{i}=zeros(sx,k); 
        pv=1./(1+exp(theta(1)+x*(mu(i)+sig(i)*qa(1))));
        p1mp=pv.*(1-pv);
        if (nargout > 1)
          dcdt{i}(:,1)=-qw(1)*p1mp;
          dcdt{i}(:,2+(i-1)*2)=-qw(1)*p1mp.*x;
          dcdt{i}(:,3+(i-1)*2)=-sig(i)*qa(1)*qw(1)*p1mp.*x;
        end
        cpv(:,i)=qw(1)*pv; 

        for j=2:nqp
          pv=1./(1+exp(theta(1)+x*(mu(i)+sig(i)*qa(j))));
          p1mp=pv.*(1-pv);
          if (nargout > 1)
            dcdt{i}(:,1)=dcdt{i}(:,1)-qw(j)*p1mp;
            dcdt{i}(:,2+(i-1)*2)=dcdt{i}(:,2+(i-1)*2)-qw(j)*p1mp.*x;
            dcdt{i}(:,3+(i-1)*2)=dcdt{i}(:,3+(i-1)*2)-sig(i)*qa(j)*qw(j)*p1mp.*x;
          end
          cpv(:,i)=cpv(:,i)+qw(j)*pv;
        end
        mcp=mcp+p(i)*cpv(:,i);
        if (nargout > 1)
          for j=1:1+2*nt
           dmcpdt(:,j)=dmcpdt(:,j)+p(i)*dcdt{i}(:,j);
          end
          for j=1:nt-1
            dmcpdt(:,1+nt*2+j)=dmcpdt(:,1+nt*2+j)+dpdt(i,j)*cpv(:,i);
          end
        end
      end
 
      if (nargout > 1)
        varargout{2}=dmcpdt;
        varargout{3}=p;
        varargout{4}=[mu'; sig'];
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
