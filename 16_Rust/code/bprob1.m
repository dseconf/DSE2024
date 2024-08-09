% bprob1.m finite mixture of binary logits
%          This version is similar to bprob.m except both intercept and slope (coefficient of x) are allowed
%          to be type-specific. 
%          John Rust, Georgetown University, July 2024

function varargout=bprob1(x,theta,T);

  % x is a matrix of covariates with sx rows and nx columns, but here we assume nx=1. This needs to be generalized in the future
  % varargout{1} is the CCP, i.e. finite mixture probability
  % varargout{2} is the gradient of the CCP with respect to the parameters defining the finite mixture
  % varargout{3} is the hessian of the CCP with respect to parameters defining the finite mixture
  % varargout{4} is the vector of type probabilities for types t=1,...,nt
  % varargout{5} is the vector of coefficients for each type, i.e. a vector of dimension 2*nt, i.e. first 2*nt components of theta
  % varargout{6} is the gradient of the type probabilities with respect to the parameters that define the (multinomial logit)
  % varargout{7} is the hessian of the type probabilities with respect to the parameters that define the (multinomial logit)
  % varargout{8} is the the matrix of CCPs for each of the nt type probabilities 


  k=numel(theta);
  sx=size(x,1);
  nt=(k+1)/3;

  nobs=numel(x);

  if (nt == 1)  % one type (homogeneous) specification

    bpv=1./(1+exp(theta(1)+x*theta(2)));
    if (nargout > 1)
       dbpv=zeros(sx,k);
       dbpv(:,1)=-bpv.*(1-bpv);
       dbpv(:,2)=-x.*bpv.*(1-bpv);
       varargout{2}=dbpv;
       if (nargout > 2)
         hbpv=zeros(sx,k,k);
         vvec=bpv.*(1-2*bpv).*(1-bpv);
         hbpv(:,1,1)=vvec;
         hbpv(:,1,2)=x.*vvec;
         hbpv(:,2,1)=hbpv(:,1,2);
         hbpv(:,2,2)=x.*x.*hbpv(:,1,1); 
         varargout{3}=hbpv;
         if (nargout > 3)
           varargout{4}=1;
           varargout{5}=theta; 
           varargout{6}=0;
           varargout{7}=0;
         end
       end
    end

    varargout{1}=bpv;

    if (nargout > 7)
      varargout{8}=bpv;
    end

    if (T > 1)

        % In this case the first output argument is a binomial distribution of possible 0/1 choices
        % evaluated at a vector of probabilities corresponding to a vector x of covariates. Thus, the
        % return in this case is a matrix with dim(x) rows and T+1 columns where in the binomial the possible outcomes range
        % for t=0,1,...,T
        % The second return is the gradient of the choice probability, which is a 3-dimensional array whose 1st dimension is x,
        % second dimension is t (from 0 to T) and 3rd dimension is the number of parameters, which is 2 in this case

        binprobs=zeros(sx,T+1);
        dbinprobs=zeros(sx,T+1,2);
        hbinprobs=zeros(sx,T+1,2,2);
        if (nargout > 2)
          cp_dbinprobs=zeros(sx,4);
          ct=0;
          for i=1:2
             for j=1:2
               ct=ct+1;
               cp_dbinprobs(:,ct)=dbpv(:,i).*dbpv(:,j);
             end
          end               
        end
        for t=0:T
          binprobs(:,t+1)=binopdf(t,T,bpv);
          if (nargout > 1)
            tmp=t*binprobs(:,t+1)./bpv;
            tmp1=(T-t)*binprobs(:,t+1)./(1-bpv);
            dbinprobs(:,t+1,:)=(tmp-tmp1).*dbpv;
            if (nargout > 2)
              hbinprobs(:,t+1,:)=(tmp-tmp1).*hbpv(:,:);
              tmp2=(t-1)*tmp./bpv;
              tmp3=(T-t)*tmp./(1-bpv);
              tmp4=(T-t-1)*tmp1./(1-bpv);
              hbinprobs(:,t+1,:)=squeeze(hbinprobs(:,t+1,:))+(tmp2-2*tmp3+tmp4).*cp_dbinprobs(:,:);
            end
          end
        end

        varargout{1}=binprobs;
        if (nargout > 1)
          varargout{2}=dbinprobs;
          if (nargout > 2)
             varargout{3}=hbinprobs;
          end
        end


    end

  else % nt > 1 types

      p=zeros(nt,1);
      cpv=zeros(sx,nt);
      mcp=zeros(sx,1);
      if (nargout > 1)
       dpdt=zeros(nt,nt-1);
       dmcpdt=zeros(sx,k);
      end

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

      dcdt=zeros(sx,2*nt,nt);  % stores the gradient of each type-specific CCP with respect all 2*nt type parameters

      for i=1:nt
        pv=1./(1+exp(theta(2*(i-1)+1)+x*theta(2*i)));
        if (nargout > 1)
          p1mp=pv.*(1-pv);
          dcdt(:,1+2*(i-1),i)=-p1mp;
          dcdt(:,2*i,i)=-p1mp.*x;
        end
        cpv(:,i)=pv; 

        mcp=mcp+p(i)*cpv(:,i);
        if (nargout > 1)
          dmcpdt(:,1+2*(i-1))=p(i)*dcdt(:,1+2*(i-1),i);
          dmcpdt(:,2*i)=p(i)*dcdt(:,2*i,i);
          for j=1:nt-1
            dmcpdt(:,2*nt+j)=dmcpdt(:,2*nt+j)+dpdt(i,j)*cpv(:,i);
          end
        end

      end

      if (nargout > 3)
        varargout{4}=p;
        varargout{5}=dpdt;
        varargout{6}=hpdt;
        varargout{7}=theta(1:2*nt);
        varargout{8}=cpv;
      end

      varargout{1}=mcp;

      if (nargout > 1)
        varargout{2}=dmcpdt;
      end

      if (nargout > 2)

         hmcpdt=zeros(sx,k,k);

         tmp1=zeros(sx,nt-1,nt-1);
         tmp2=zeros(sx,nt-1);
         tmp3=zeros(sx,2,nt-1,nt);

         for i=1:nt

           hbpv=zeros(sx,2,2);
           bpv=cpv(:,i);
           vvec=bpv.*(1-2*bpv).*(1-bpv);
           hbpv(:,1,1)=vvec;
           hbpv(:,1,2)=x.*vvec;
           hbpv(:,2,1)=hbpv(:,1,2);
           hbpv(:,2,2)=x.*x.*hbpv(:,1,1); 

           hbpv=p(i)*hbpv;

           % now fill in this into the appropriate blocks of hmcpdt

           hmcpdt(:,1+2*(i-1):2*i,1+2*(i-1):2*i)=hbpv;

           tmp=hpdt(:,:,i);
           tmp1(:,:)=bpv.*tmp(:)';                      
           hmcpdt(:,2*nt+1:end,2*nt+1:end)=hmcpdt(:,2*nt+1:end,2*nt+1:end)+tmp1;

           tmp3(:,1,:,i)=dcdt(:,1+2*(i-1),i).*dpdt(i,:);
           tmp3(:,2,:,i)=dcdt(:,2*i,i).*dpdt(i,:);
           tmp4=squeeze(tmp3(:,:,:,i));
           hmcpdt(:,1+2*(i-1):2*i,2*nt+1:end)=tmp4;
           if (nobs == 1)
             hmcpdt(:,2*nt+1:end,1+2*(i-1):2*i)=tmp4';
           else
             hmcpdt(:,2*nt+1:end,1+2*(i-1):2*i)=permute(tmp4,[1 3 2]);
           end

         end

         varargout{3}=hmcpdt;

         if (nargout > 3)
           varargout{4}=p;
           varargout{5}=theta(1:2*nt); 
           varargout{6}=dpdt;
           varargout{7}=hpdt;
         end

      end

      if (T > 1)

        % In this case the first output argument is a binomial distribution of possible 0/1 choices
        % evaluated at a vector of probabilities corresponding to a vector x of covariates. Thus, the
        % return in this case is a matrix with dim(x) rows and T+1 columns where in the binomial the possible outcomes range
        % for t=0,1,...,T
        % The second return is the gradient of the choice probability, which is a 3-dimensional array whose 1st dimension is x,
        % second dimension is t (from 0 to T) and 3rd dimension is the number of parameters, which is k in this case

        binprobs=zeros(sx,T+1);
        dbinprobs=zeros(sx,T+1,k);
        hbinprobs=zeros(sx,T+1,k,k);

        % calculate outer product of gradients of dmcpdt

        if (nargout > 2)
          cp_dbinprobs=zeros(sx,k);
          ct=0;
          for i=1:k
             for j=1:k
               ct=ct+1;
               cp_dbinprobs(:,ct)=dmcpdt(:,i).*dmcpdt(:,j);
             end
          end               
        end

        for t=0:T
          binprobs(:,t+1)=binopdf(t,T,mcp);
          if (nargout > 1)
            tmp=t*binprobs(:,t+1)./mcp;
            tmp1=(T-t)*binprobs(:,t+1)./(1-mcp);
            dbinprobs(:,t+1,:)=(tmp-tmp1).*dmcpdt;
            if (nargout > 2)
              hbinprobs(:,t+1,:)=(tmp-tmp1).*hmcpdt(:,:);
              tmp2=(t-1)*tmp./mcp;
              tmp3=(T-t)*tmp./(1-mcp);
              tmp4=(T-t-1)*tmp1./(1-mcp);
              hbinprobs(:,t+1,:)=squeeze(hbinprobs(:,t+1,:))+(tmp2-2*tmp3+tmp4).*cp_dbinprobs(:,:);
            end
          end
        end

        varargout{1}=binprobs;

        if (nargout > 1)
          varargout{2}=dbinprobs;
          if (nargout > 2)
             varargout{3}=hbinprobs;
          end
        end

      end

  end
