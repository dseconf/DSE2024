% gendata.m: generates simulated data from the mixed binary logit model
%            John Rust, Georgetown University, July 2024
%
% nobs       number of observations
% T          panel dimension: number of periods each agent is observed
% model_params  a matrix containing the parameters of the "true model" used
%               to generate the simulated data
% true_p        a vector of mixture probabilities determiningg the probability
%               of each distinct type of consumer used to generate the simulations
% mixture_type  a string that is either 'continuous' if the true heterogeneity is 
%               continuously distributed with a bivariate normal distribution for
%               the intercept and slope coefficient of x for each different type of
%               consumer, or 'discrete' if the heterogeneity is just a finte mixture
%               where all individuals of a given type have the same intercept and
%               slope parameters.

function [y,x,tv]=gendata(nobs,T,model_params,true_p,mixture_type)

   nt=numel(true_p);
   csp=cumsum(true_p);

   y=zeros(nobs,T);
   x=2*rand(nobs,T);     % uniformly distributed x data on the [0,2] interval
   tv=zeros(nobs,T);

   if (strcmp(mixture_type,'continuous'))
      model_params=reshape(model_params(1:5*nt),5,nt);
   else
      model_params=reshape(model_params(1:2*nt),2,nt);
   end

   for i=1:nobs    % loop through the data generating randome coefficients

      typeindex=min(find(rand(1,1) <= csp));
      tv(i)=typeindex;

      % now generate the random coefficients for the logit

      if (strcmp(mixture_type,'continuous'))

          % generate beta coefficients from a bivariate normal
          % The 5 parameters for the bivariate normal are determined as follows
          % beta(1), the intercept coefficient is univariate normal with a  mean of params(1)
          %          and standard deviation exp(params(2)/2) 
          % beta(2)=params(3)+theta(5)*beta(1)+epsilon(2) where epsilon(2) is univariate normal
          %         with mean 0 and standard deviation exp(params(4)/2).
          %         Thus if params(5) is non-zero then beta(1) and beta(2) are correlated with covariance equal to
          %         cov(beta(1),beta(2))=params(5)*exp(params(2))

          params=model_params(:,typeindex);

          beta_coeff=zeros(2,1);
          beta_coeff(1)=params(1)+randn(1,1)*exp(params(2)/2);
          beta_coeff(2)=params(3)+params(5)*beta_coeff(1)+randn(1,1)*exp(params(4)/2);

          for t=1:T

            px=1/(1+exp(beta_coeff(1)+x(i,t)*beta_coeff(2)));   

            if (rand(1,1)<=px)
               y(i,t)=1;
            end

          end  % end of loop over number of time periods T each person is observed

      else

          % generate beta coefficients of the logit from a finite mixture

          params=model_params(:,typeindex);
        
          for t=1:T 

             px=1/(1+exp(params(1)+x(i,t)*params(2)));   

             if (rand(1,1)<=px)

                y(i,t)=1;

             end

          end  % end of loop over number of time periods T each person is observed

      end  % end of if branch over continuous or discrete heterogeneity

   end  % end of loop over individuals in simulated sample

end

