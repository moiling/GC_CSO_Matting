function [fitness, est_alpha] = CSO_CostFunc_all_fitness( x,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist )
%CCPSO_COSTFUNC Summary of this function goes here
%   Detailed explanation goes here
    if size(x,1)>1
        fitness = zeros(size(x,1),size(U_rgb,1));
        est_alpha = zeros(size(x,1),size(U_rgb,1));
        for i = 1:size(x,1)
            x_i = reshape(x(i,:),[size(x,2)/2,2]);
            [ fitness_i,est_alpha_i] = CostFuncRound( x_i,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist);
            fitness(i, :) = reshape(fitness_i, 1, size(fitness_i, 1));
            est_alpha(i,:) = est_alpha_i;
        end
    else
        x = reshape(x', [length(x)/2,2]);
        [fitness, est_alpha] = CostFuncRound(x, F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist);
    end
    
end

