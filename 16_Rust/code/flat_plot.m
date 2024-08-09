% flat_plot.m plots an illustrative example of a problem with poorly identified minimum

x=(-2:.01:2)';

y=-1./(1+exp(+12-2*x.^2));

plot(x,y,'-r','Linewidth',2);
title('Example of a poorly identified parameter via maximum likelihood');
xlabel('Parameter \theta');
ylabel('Likelihood function');

