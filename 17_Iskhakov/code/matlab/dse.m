% This script runs Matlab assignments from the lecture on EGM and DC-EGM

classdef dse
methods (Static)

%% Set up the problem
function init()
  addpath('utils');
  m=model_deaton %create the model objects
  assignin('base','m',m); % save the model in the workspace
end

%% Solve the model with VFI
function solve_vfi()
  m=evalin('base','m'); % grab the model from the workspace
  fprintf('\nSolving %s with value function iterations:\n',m.label)
  tic
  m.solve_vfi;
  t=toc;
  fprintf('Done in %s\n',ht(t))
  m.plot('policy');
end
%%

%% Solve the model with EGM
function solve_egm(K)
  if ~exist('K','var')
    K=100;
  end
  m=evalin('base','m'); % grab the model from the workspace
  fprintf('\nSolving %s with EGM %d times:\n',m.label,K)
  tic
  for i=1:K
  fprintf('.');  
  m.solve_egm;
  end
  t=toc;
  fprintf('\nDone in %s, on avarage %s per run\n',ht(t),ht(t/K))
  m.plot('policy');
end
%%

%% Simulate some data
function simulate
  m=evalin('base','m'); % grab the model from the workspace
  m.solve_egm;
  m.nsims=100;
  m.sim;
  m.sims
  m.plot('sim wealth0')
  % m.plot('sim consumption')
end
%%

%% Simulate flat consumption paths
function make_it_flat
  m=evalin('base','m'); % grab the model from the workspace
  m.df=1/(1+m.r);
  m.sigma=0;
  m.init=[30 35];
  m.nsims=2;
  % m.solve_egm;
  m.solve_vfi;
  m.sim;
  m.plot('sim consumption');
end
%%

%% EGM with value functions
function solve_egm_vf
  m=evalin('base','m'); % grab the model from the workspace
  m.inc0=2.5;
  m.solve_egm(true);
  m.plot('solution');
  m.plot('value');
end


%% Flat simulated consumption path using retirement model without taste shocks
function retirement_make_it_flat
  m2=model_retirement;
  m2.ngridm=500;
  m2.df=1/(1+m2.r); %flat consumption hopefully
  m2.sigma=0;
  m2.lambda=eps; %no EV taste shocks
  m2.nsims=2;
  m2.init=[5 20];
  tic
  m2.solve_dcegm;
  t=toc;
  fprintf('Retirement model solved with DC-EGM in %s\n',ht(t));
  m2.plot('policy')
  m2.plot('value')
  m2.plot('prob_work')
  m2.sim;
  m2.plot('sim consumption');
  ylim=get(gca,'Ylim');
  set (gca,'Ylim',[min(min(m2.sims.consumption))-ylim(2)+max(max(m2.sims.consumption)),ylim(2)]);
  assignin('base','m2',m2); % save the model in the workspace
end
%%

%% Retirement model with taste shocks
function retirement_shocks
  m2=evalin('base','m2'); % grab the model from the workspace
  m2.sigma=0.35;
  m2.lambda=0.2; %some EV taste shocks
  tic
  m2.solve_dcegm;
  t=toc;
  fprintf('Retirement model solved with DC-EGM in %s\n',ht(t));
  % m2.plot('policy');
  % m2.plot('value');
  m2.plot('prob_work');
  m2.nsims=100;
  m2.sim;
  % m2.plot('sim wealth0 consumption income retirement_age')
  m2.plot('sim wealth1 consumption retirement_age')
  fprintf('Simulation plots for retirement model produced\n')
end

end %methods
end %classdef
