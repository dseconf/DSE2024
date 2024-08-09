% nllf.m: (negative of) the mixed logit log likelihood function
%         John Rust, Georgetown University, February, 2018

  function varargout=nllf(y,x,theta,prob);

    if (nargout > 1);
     [llfv,dllfv]=prob(x,theta);
     varargout{2}=-sum(y.*dllfv./llfv-(1-y).*dllfv./(1-llfv));
    else;
     llfv=prob(x,theta);
    end;

    varargout{1}=-sum(y.*log(llfv)+(1-y).*log(1-llfv));



