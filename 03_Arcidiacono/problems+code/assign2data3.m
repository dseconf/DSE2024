%creating data for problem set 2 question 3

%rates of state change

q=.5;
lambda=1;

%euler's constant
eul=.5772156649;

%Maximum number of moves
Nmax=20000;

%time period
T=10000;

%states

S=[0;1];

%coefficients
u=[.3;-3];
c=2;

%solving a fixed point problem to get the value functions

Vinit=[10;10];

V=[0;0];

rho=.05;

j=1;
while max(abs(V-Vinit))>.000001
    
    Vinit=V;
    V=(u+q*[Vinit(2);Vinit(1)]+lambda*(log(exp(Vinit)+1)+eul))./(q+lambda+rho);
   
    j=j+1;
end

%drawing when each ven occurred

movetime=-(log(1-rand(Nmax,1)))./(lambda+q);
movetimec=cumsum(movetime);

%only taking moves that occurred before T

mt=movetimec(movetimec<T);
nm=size(mt,1);

%probablity of taking an action given the state and incumbency status
P=[exp(V-c)./(1+exp(V-c));exp(V)./(1+exp(V))];

%probability that a move was from nature

ncprob=q/(q+lambda);

%getting nature's moves

Naturemove=rand(nm,1)<ncprob;
Naturemovec=cumsum(Naturemove);

%having nature's move change the state

State=[(Naturemovec./2)==round(Naturemovec./2)];

%getting the moves of the player and tracking incumbency status

IState=zeros(nm,1);
LIState=zeros(nm+1,1);
Draw=rand(nm,1);
m=1;
while m<nm+1;
    if Naturemove(m)==0;
        IState(m)=(Draw(m,1)<P(1+State(m)+2*LIState(m)));
    else
        IState(m)=LIState(m);
    end
    LIState(m+1)=IState(m);
    m=m+1;
end

save dataassign23 Naturemove IState State mt


        




