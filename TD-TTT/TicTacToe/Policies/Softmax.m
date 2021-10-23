function num = Softmax(Grades)
%EPSGREEDY Softmax policy
% Grades    - The critic grades for each possible action
% num       - The chosen action's index

% TODO: Implement the softmax policy
beta = 20;

% beta = length(Grades)*beta;

%calculating the denominator of the formula.
sum_a = 0;
for i = 1:length(Grades)
    
    sum_a = sum_a + exp(beta*Grades(i));
end

%calculating the counter of the formula.
prob_grades = [];
for i = 1:length(Grades)
    prob_grades(i) = exp(beta*Grades(i));
end

%full combine of the formula.
prob_grades_final = prob_grades.*(1/sum_a);

%choosing num according to probability.
cum_prob = cumsum(prob_grades_final);
prob = rand;
idx = find(cum_prob<prob);

num = length(idx)+1;



end
