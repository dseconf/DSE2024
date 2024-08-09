function [probtypev]=probtype(n_types,theta,type);

    d=1;
    probtypev=zeros(n_types,1);
    probtypev(1)=1;
    if (strcmp(type,'discrete'));
      for i=1:n_types-1;
        probtypev(i+1)=exp(theta(1+n_types+i));
        d=d+probtypev(i+1);
      end; 
    else;
      for i=1:n_types-1;
        probtypev(i+1)=exp(theta(1+2*n_types+i));
        d=d+probtypev(i+1);
      end; 
    end;
    probtypev=probtypev/d;
