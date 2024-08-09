% discrete_cdf.m computes the cdf of a discrete mixture, i.e. a CDF with a finite number of mass points
%                John Rust, Georgetown University, July 2024


function [cdfv]=discrete_cdf(x,xswitch,ord,dcdf)

  if (xswitch)  % find the probability of being less than or equal to x

     if (x < min(ord))
 
       cdfv=0;

     elseif (x > max(ord))

       cdfv=1;

     else

       ind=find(ord <=x);

       cdfv=sum(dcdf(ind));

     end
       
  else   % find the probability of being strictly less than x

     if (x < min(ord))
 
       cdfv=0;

     elseif (x > max(ord))

       cdfv=1;

     else

       ind=find(ord <x);

       if (numel(ind))

         cdfv=sum(dcdf(ind));

       else

         cdfv=0;

       end

     end

  end
