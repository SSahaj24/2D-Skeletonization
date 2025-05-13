%% Weighted Nearest Neighbor
% Samik Banerjee, May 4, 2025
% Input: Data, Query, # Nearest Neighbor
% Output: Weights, Index of K-nearest neighbor of Query


function [W, Idx] = wiNN_3D(X,Y, k)
    % X = density_PMD_R;
    % k = 5;
    [Idx, Dist] = knnsearch(X, Y, 'K', k+1);
    
    R_k1 = Dist(:, k+1);
    VT = ones(size(Dist));
    V1 = rdivide(Dist,R_k1);
    V = VT./V1;
    
    V_sum = sum(V, 2);
    
    W = rdivide(V, V_sum);
    W(isnan(W)) = 1/(k+1);

end






















