%shell program estimating problem set 2, problem 3
clear
load dataassign23

%defining time period, discount rate, and euler's constant
T=10000;
rho=0.05;
eul=.5772156649;

%creating an indicator for when the agent had the right to move
IMove=1-Naturemove;

%creating a vector for lagged incumbency status
LIState=[0;IState(1:end-1)];

%counting the number of times each action was taken in each state and
%incumbency status combination

Y=zeros(1,8);

j=0;
while j<2
    s=0;
    while s<2
        flag=IMove.*(LIState==j).*(State==s);
        Y(s+2*j+1)=sum(IState(flag==1));
        Y(s+2*j+5)=sum(1-IState(flag==1));
        s=s+1;
    end
    j=j+1;
end

%calculating the rate at which nature moves
qest=sum(Naturemove)./T;

%calculating the move opportunity rate
lambdaest=sum(1-Naturemove)./T;
tic 
[b,like,e,o,g,h]=fminunc('ctimelike',[0;0;0],[],Y,qest,lambdaest,rho,eul);
toc
[b sqrt(diag(inv(h)))]