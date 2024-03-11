function [lambdaTH] = thresholdCSK(var_1,var_0, mean_1,mean_0)
lambdaTH = (1/(var_1-var_0))*((var_1*mean_0)-(var_0*mean_1)+...
       (sqrt(var_1)*sqrt(var_0))*sqrt(((mean_1-mean_0)^2)+2*(var_1-var_0)*log(sqrt(var_1)/(sqrt(var_0)))));
end