% mixed_normal_cdf.m: CDF for a finite mixture of a normals
%                     John Rust, Georgetown University, July 2024

% Inputs 
% mixture_prob a vector of dimension n_types containing the probabilities of the mixture components
% mixture_params a matrix of dimension 2 x n_types with the mean and std deviation of the normal mixture components
% x              a vector of values to evaluate the mixture probability at

function  [mcdf]=mixed_normal_cdf(mixture_prob,mixture_params,x)

n_types=numel(mixture_prob);
nx=numel(x);
mcdf=zeros(nx,1);

for t=1:n_types
  mcdf=mcdf+mixture_prob(t)*cdf('norm',x,mixture_params(1,t),mixture_params(2,t));
end


end
