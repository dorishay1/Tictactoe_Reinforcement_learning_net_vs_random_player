function InitializeNet()
%INITIALIZENET Initialize the network's weights and activation functions

% TODO: The current implementation is for a linear perceptron. Change it.

mpath = strrep(which(mfilename),[mfilename '.m'],'');
addpath([mpath 'ActivationFunctions']);

% Set the network's dimensions
N = [10,30,30,1];
L = length(N) - 1;

% Create a new network
global Net;
Net = [];
Net.W = cell(1, L);
Net.g = cell(1, L);

%Xavier
mu = 0;
M = 2/(N(1)+N(end));
% For each layer
for l = 1:L
    
    % Initialize the layer's weights
%     Net.W{l} = unifrnd(-0.001, 0.001, [N(l + 1), N(l) + 1]);
    Net.W{l} = M.*randn(N(l + 1),N(l) + 1) + mu;
    
    % Set the layer's activation functions
    if l < L
        Net.g{l} = @ReLU;
    elseif l == L
        Net.g{l} = @Sigmoid;
        
    end
    
    SaveNet();
    
end
