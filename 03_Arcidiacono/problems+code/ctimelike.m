
function [like]=ctimelike(b,Y,q,lambda,rho,eul)

global Vtemp 

%parsing the parameters

bu=b(1:2);
bc=b(3);

%initializing the value function

V=zeros(2,1);
Vinit=ones(2,1);

%solving a fixed point problem to get the value function for particular
%values of the parameters

j=1;
while max(abs(V-Vinit))>.00000000001
    
    Vinit=V;
    V=(bu+q*[Vinit(2);Vinit(1)]+lambda*(log(exp(Vinit)+1)+eul))./(q+lambda+rho);
   
    j=j+1;
end

Vtemp=V;

%calculating the probablities of taking the actions given the states

P=[exp(V-bc)./(1+exp(V-bc));exp(V)./(1+exp(V))];

P=[P;1-P];

%weighting the log probabilities by the number of times the action was
%taken in a particular state
like=-Y*log(P);
