function TDTrain(S, r, eta)
% Train the TD system
% S - The states' history for a given trial
% r - the reward given for each state in the given trial

gamma = 0.97 ;


% Load the network's data if needed
global Net;
LoadNet();

% TODO: Implement TD(0). It is highly recommended to read the documentation
% of TDEvaluate first.

for i=1:size(S,2)
    [v_current, grad] = TDEvaluate(S(:,i));
    
    if i < size(S,2)
        [v_next,~] = TDEvaluate(S(:,i+1));
        
    else
        v_next = 0;
        
        
    end
    
    
  
    for l = 1:size(Net.W,2)
        
        delta_t = r(i) + gamma*v_next - v_current;
        
        temp_grad = reshape(grad{l}(1, :, :),[size(grad{l},2),size(grad{l},3)]);
        delta_W = eta*delta_t*v_current.*temp_grad;
        
        Net.W{l}  = Net.W{l}+ delta_W;
        
    end
end

