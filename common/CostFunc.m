function [fitness, est_alpha] = CostFunc(x, F_rgb, B_rgb, U_rgb, F_s, B_s, U_s, F_mindist, B_mindist, out_all)

    if ~exist('out_all', 'var') || isempty(out_all)
        out_all = false;
    end
    
    if size(x, 1) > 1
        if ~out_all
            fitness = zeros(size(x, 1), 1);           
        else
            fitness = zeros(size(x, 1), size(U_rgb, 1));
        end
        est_alpha = zeros(size(x, 1), size(U_rgb, 1));
        
        for i = 1:size(x, 1)
            x_i = reshape(x(i, :), [size(x, 2) / 2, 2]);
            [fitness_i, est_alpha_i] = CostFuncRound(x_i, F_rgb, B_rgb, U_rgb, F_s, B_s, U_s, F_mindist, B_mindist);
            est_alpha(i,:) = est_alpha_i;
            
            if ~out_all
                fitness(i) = sum(fitness_i);
            else
                fitness(i, :) = reshape(fitness_i, 1, size(fitness_i, 1));               
            end
        end
    else
        x = reshape(x', [length(x)/2, 2]);
        [fitness, est_alpha] = CostFuncRound(x, F_rgb, B_rgb, U_rgb, F_s, B_s, U_s, F_mindist, B_mindist);
        
        if ~out_all
            fitness = sum(fitness);
        end
    end
    
end

