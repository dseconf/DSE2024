% bprob2.m finite mixture of binary logit (panel data version)s
%          This version is similar to bprob1.m except both intercept and slope (coefficient of x) are allowed
%          to be type-specific. It computes the conditional probability f(y|x,theta) as a panel mixture
%          where y=(y_1,...,y_T) are binary 0/1 outcomes and x=(x_1,...,x_T) are covariates for T observations for
%          a single individual or "unit" and conditional on theta, outcomes are independent given x 
%          John Rust, Georgetown University, July 2024

function varargout=bprob2(varargin)

  % y is a vector of binary outcomes, its length, T, is the length of the panel
  % x is a vector of univariate covariates, also of length T.
  % varargout{1} is the conditional probability f(y|x,theta), i.e. finite mixture over conditional densities for the panel
  % varargout{2} is the gradient of f(y|x,theta) with respect to theta
  % varargout{3} is the hessian of f(y|x,theta) with respect to theta
  % varargout{4} is the vector of type probabilities for types t=1,...,nt
  % varargout{5} is the vector of coefficients for each type, i.e. a vector of dimension 2*nt, i.e. first 2*nt components of theta
  % varargout{6} is the gradient of the type probabilities with respect to the parameters that define the (multinomial logit)
  % varargout{7} is the hessian of the type probabilities with respect to the parameters that define the (multinomial logit)

  y=varargin{1};
  x=varargin{2};
  theta=varargin{3};

  T=numel(y);

  if (numel(x) ~= T)

      fprintf('Error: vector of covariates does not have correct panel dimension as y, %i\n',T);
      varargout{1}=NaN;
      return;

  end

  k=numel(theta);
  sx=size(x,1);
  nt=(k+1)/3;

  ldt=zeros(nt,1);    % log(f(y|x,theta_n)) for n=1,...,nt
  gldt=zeros(nt,2);   % gradient of log(f(y|x,theta_n)) with respect to theta_n (2x1) for n=1,...,nt
  hldt=zeros(nt,2,2); % hessian of log(f(y|x,theta_n)) with respect to theta_n (2x1) for n=1,...,nt
  fyxt=zeros(nt,1);   % computes f(y_1,...,y_T|x_1,...,x_T,theta_k) for k=1,...,nt
  p=zeros(nt,1);      % stores the type probabilities (mixture probabilities)
  cpv=zeros(T,nt);    % stores the binary logit probabilities p(x_t,theta_k), t=1,...,T, k=1,...,nt
  
  mcp=0;              % the overall mixed choice probability p*cpv=f(y_1,...,y_T|x_1,...,x_T)
  if (nargout > 1)
  dcpdt=zeros(T,2,nt);% stores the gradient of p(x,theta_t) with respect to the 2 parameters in theta_t for all nt types t
    if (nargout > 2)
      hcpdt=zeros(T,2,2,nt);  % stores the hessian of p(x,theta_t) with respect to the 2 parameters in theta_t for all nt types t
    end
  end

  if (nargout > 1)
    dpdt=zeros(nt,nt-1);  % gradients of the type probabilities with respect to their parameter values
    dmcpdt=zeros(k,1);
    if (nargout > 2)
      hmcpdt=zeros(k,k);
    end
  end

  % compute the type probabilities (mixture probabilities) and their gradients/hessian with respect to the logit parameters

  if (nargin < 4)

      for i=1:nt
        if (i ==1)
           p(i)=1;
        else
           p(i)=exp(theta(2*nt+i-1));
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

      if (nargout > 2)  % compute the hessian matrix of the type probabilities with respect to the parameters defining
                        % these type specific probabilities

          hpdt=zeros(nt-1,nt-1,nt);  % 3rd array element indexes each probability, from 1 to nt. First two dimensions of the
                                     % array store the hessian of each of the nt probabilities with respect to the nt-1 parameters
                                     % that determine this probabilities

          for t=1:nt-1     % enter the diagonals for p1, the hessian of the probability of type 1

              hpdt(t,t,1)=p(1)*p(t+1)*(2*p(t+1)-1);

          end

          % now fill in the off-diagonal elements recursively

          for t=1:nt-1

              for tp=t+1:nt-1

                 hpdt(t,tp,1)=2*p(1)*p(t+1)*p(tp+1);
                 hpdt(tp,t,1)=hpdt(t,tp,1);

              end

          end

          % now compute hessians for the remaining probabilities p(2),...,p(nt)
          % with respect to parameters 1,...,nt-1

          for t=2:nt

              % do the diagonals

              for tp=1:nt-1

                if (tp == t-1)   % "lagged diagonals"

                  hpdt(tp,tp,t)=p(t)*(1-p(t))*(1-2*p(t));

                else

                  hpdt(tp,tp,t)=p(t)*p(tp+1)*(2*p(tp+1)-1);

                end

              end

              % do the off-diagonals

              for tp=1:nt-1

                  for tpp=tp+1:nt-1 

                     if (tp == t-1)

                     hpdt(tp,tpp,t)=p(t)*p(tpp+1)*(2*p(t)-1);
                     hpdt(tpp,tp,t)=hpdt(tp,tpp,t);
                     hpdt(tpp,tp,t)=hpdt(tp,tpp,t);
                 
                     else

                        if (tpp == t-1)

                            hpdt(tp,tpp,t)=p(tp+1)*p(t)*(2*p(t)-1);
                            hpdt(tpp,tp,t)=hpdt(tp,tpp,t);

                        else

                            hpdt(tp,tpp,t)=2*p(t)*p(tp+1)*p(tpp+1);
                            hpdt(tpp,tp,t)=hpdt(tp,tpp,t);

                        end

                     end

                  end

              end
 
          end

      end
 
  else  % if these are provided as inputs 4,5 and 6 of bprob2, then don't recompute them

     p=varargin{4};
     dpdt=varargin{5};
     hpdt=varargin{6};

  end

  % compute the type-specific binary logit probabilities

  for i=1:nt

     cpv(:,i)=1./(1+exp(theta(2*(i-1)+1)+x*theta(2*i)));

     if (nargout > 1)
       p1mp=cpv(:,i).*(1-cpv(:,i));
       dcpdt(:,1,i)=-p1mp;
       dcpdt(:,2,i)=-p1mp.*x;
       if (nargout > 2)
         vvec=p1mp.*(1-2*cpv(:,i));
         hcpdt(:,1,1,i)=vvec;
         hcpdt(:,1,2,i)=x.*vvec;
         hcpdt(:,2,1,i)=hcpdt(:,1,2,i);
         hcpdt(:,2,2,i)=x.*x.*vvec;
       end
     end

     fyxt(i)=prod(cpv(:,i).*y+(1-cpv(:,i)).*(1-y));
     mcp=mcp+p(i)*fyxt(i);

     if (nargout > 1)

       tmp=(y./cpv(:,i)-(1-y)./(1-cpv(:,i)))'*dcpdt(:,:,i);
       dmcpdt(1+2*(i-1):2*i)=p(i)*fyxt(i)*tmp;
       dmcpdt(2*nt+1:end)=dmcpdt(2*nt+1:end)+fyxt(i)*dpdt(i,:)';

       if (nargout > 2)

          hmcpdt(2*nt+1:end,2*nt+1:end)=hmcpdt(2*nt+1:end,2*nt+1:end)+squeeze(hpdt(:,:,i))*fyxt(i);          
          hmcpdt(1+2*(i-1):2*i,2*nt+1:end)=dmcpdt(1+2*(i-1):2*i)*dpdt(i,:)/p(i); 
          hmcpdt(2*nt+1:end,1+2*(i-1):2*i)=hmcpdt(1+2*(i-1):2*i,2*nt+1:end)';

          tmp2=squeeze(sqrt(y./(cpv(:,i).^2)+(1-y)./(1-cpv(:,i)).^2).*dcpdt(:,:,i));
          if (T > 1)
            tmp3=squeeze(sum((y./cpv(:,i)-(1-y)./(1-cpv(:,i))).*hcpdt(:,:,:,i)));
          else
            tmp3=squeeze(((y./cpv(:,i)-(1-y)./(1-cpv(:,i))).*hcpdt(:,:,:,i)));
          end

          hmcpdt(1+2*(i-1):2*i,1+2*(i-1):2*i)=p(i)*fyxt(i)*(tmp'*tmp-tmp2'*tmp2+tmp3);

       end

     end
 
  end

  varargout{1}=mcp;

  if (nargout > 1)

    varargout{2}=dmcpdt;

    if (nargout > 2)

       varargout{3}=hmcpdt;

       if (nargout > 3)

          varargout{4}=p;

          if (nargout > 4)

            varargout{5}=theta(1:2*nt);

            if (nargout > 5)

               varargout{6}=dpdt;

               if (nargout > 6)

               varargout{7}=hpdt;

               end

            end

          end

       end

    end

  end
  

end
