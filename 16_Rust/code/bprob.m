% bprob.m finite mixture of binary logits
%         John Rust, Georgetown University, July 2024

function varargout=bprob(x,theta,T);

  % x is a matrix of covariates with sx rows and nx columns, but here we assume nx=1. This needs to be generalized in the future

  k=size(theta,1);
  sx=size(x,1);
  nt=k/2;

  if (nt == 1)  % one type (homogeneous) specification

    bpv=1./(1+exp(theta(1)+x*theta(2)));
    if (nargout > 1)
       dbpv=zeros(sx,k);
       dbpv(:,1)=-bpv.*(1-bpv);
       dbpv(:,2)=-x.*bpv.*(1-bpv);
       varargout{2}=dbpv;
%       if (nargout > 2)
%         m=zeros(2,2);
%         m(1,1)=sx;
%         m(2,1)=sum(x);
%         m(1,2)=m(2,1)';
%         m(2,2)=x'*x;
%         varargout{3}=bpv.*(1-2*bpv).*(1-bpv)*m;
%       end
       if (nargout > 3)
           varargout{3}=1;
           varargout{4}=theta(2); 
       end
    end

    varargout{1}=bpv;

    if (T > 1)

        % In this case the first output argument is a binomial distribution of possible 0/1 choices
        % evaluated at a vector of probabilities corresponding to a vector x of covariates. Thus, the
        % return in this case is a matrix with dim(x) rows and T+1 columns where in the binomial the possible outcomes range
        % for t=0,1,...,T
        % The second return is the gradient of the choice probability, which is a 3-dimensional array whose 1st dimension is x,
        % second dimension is t (from 0 to T) and 3rd dimension is the number of parameters, which is 2 in this case

        binprobs=zeros(sx,T+1);
        dbinprobs=zeros(sx,T+1,2);
        for t=0:T
          binprobs(:,t+1)=binopdf(t,T,bpv);
          if (nargout > 1)
          dbinprobs(:,t+1,:)=binprobs(:,t+1).*(t./bpv-(T-t)./(1-bpv)).*dbpv;
          end
        end

        varargout{1}=binprobs;
        if (nargout > 1)
          varargout{2}=dbinprobs;
        end

    end

  else % nt > 1 types

      p=zeros(nt,1);
      cpv=zeros(sx,nt);
      mcp=zeros(sx,1);
      if (nargout > 1)
       dpdt=zeros(nt,nt-1);
       dcdt=cell(nt,1);
       dmcpdt=zeros(sx,k);
      end

      for i=1:nt
        if (i ==1)
           p(i)=1;
        else
           p(i)=exp(theta(1+nt+i-1));
        end
      end;
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
        pv=1./(1+exp(theta(1)+x*theta(1+i)));
        if (nargout > 1)
          dcdt{i}(:,1)=-pv.*(1-pv);
          dcdt{i}(:,1+i)=-pv.*(1-pv).*x;
        end
        cpv(:,i)=pv; 

        mcp=mcp+p(i)*cpv(:,i);
        if (nargout > 1)
          for j=1:1+nt
           dmcpdt(:,j)=dmcpdt(:,j)+p(i)*dcdt{i}(:,j);
          end
          for j=1:nt-1
            dmcpdt(:,1+nt+j)=dmcpdt(:,1+nt+j)+dpdt(i,j)*cpv(:,i);
          end
        end
      end
 
      if (nargout > 1)
        varargout{2}=dmcpdt;
        varargout{3}=p;
        varargout{4}=theta(2:nt+1);;
      end

      varargout{1}=mcp;

      if (T > 1)

        % In this case the first output argument is a binomial distribution of possible 0/1 choices
        % evaluated at a vector of probabilities corresponding to a vector x of covariates. Thus, the
        % return in this case is a matrix with dim(x) rows and T+1 columns where in the binomial the possible outcomes range
        % for t=0,1,...,T
        % The second return is the gradient of the choice probability, which is a 3-dimensional array whose 1st dimension is x,
        % second dimension is t (from 0 to T) and 3rd dimension is the number of parameters, which is k in this case

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

