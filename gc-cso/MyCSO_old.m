function [x, bestever, F] = MyCSO(FitnessFcn, numberOfVariables, lb, ub, maxfe, initial_x, print_process, SaveProcessFcn)
% MYCSO Summary of this function goes here
%   Detailed explanation goes here
if ~exist('print_process', 'var') || isempty(print_process)
    print_process = false;
end

d = numberOfVariables;

lu = single([lb; ub]);
clear('lb', 'ub');

n = d;
initial_flag = 0;

% phi setting (the only parameter in CSO, please SET PROPERLY)
if(d >= 2000)
    phi = 0.2;
elseif(d >= 1000)
    phi = 0.1;
elseif(d >=500)
    phi = 0.05;
else
    phi = 0;
end

% population size setting
if(d >= 5000)
    population = 1500;
elseif(d >= 2000)
    population = 1000;
elseif(d >= 1000)
    population = 500;
elseif(d >= 100)
    population = 100;
else
    population = 50;
end

% initialization
% F = zeros(population,maxfe);
XRRmin = repmat(lu(1, :), population, 1);
XRRmax = repmat(lu(2, :), population, 1);
rand('seed', sum(100 * clock));

p = XRRmin;
for ii = 1:population
    p(ii, :) = XRRmin(ii, :) + (XRRmax(ii, :) - XRRmin(ii, :)) .* rand(1, d, 'single');
end

% 将之前算好的解作为初始化解，当作一个个体
if exist('initial_x', 'var') && ~isempty(initial_x) && ~initial_x == false
   p(1, :) = initial_x; 
end

% p = XRRmin + (XRRmax - XRRmin) .* rand(population, d, 'single');
clear('XRRmin', 'XRRmax');

fitness = FitnessFcn(p);

v = zeros(population, d, 'single');
bestever = 1e200;

FES = population;
F(:, 1) = [FES; fitness];
gen = 1;

% main loop
while(FES < maxfe)   

    % generate random pairs
    rlist = randperm(population);
    rpairs = [rlist(1:ceil(population / 2)); rlist(floor(population / 2) + 1:population)]';
    
    % calculate the center position
    center = ones(ceil(population / 2), 1) * mean(p);
    
    % do pairwise competitions
    mask = (fitness(rpairs(:, 1)) > fitness(rpairs(:, 2)));
    losers = mask .* rpairs(:, 1) + ~mask .* rpairs(:, 2);
    winners = ~mask .* rpairs(:, 1) + mask .* rpairs(:, 2);    

    for ii = 1:length(losers)
        v(losers(ii), :) = rand(1, d, 'single') .* v(losers(ii), :) ..., 
            + rand(1, d, 'single') .* (p(winners(ii), :) - p(losers(ii), :)) ...,
            + phi * rand(1, d, 'single') .* (center(ii) - p(losers(ii), :));
        p(losers(ii), :) = p(losers(ii), :) + v(losers(ii), :);
    end
    
    % boundary control
    for i = 1:ceil(population / 2)
        p(losers(i), :) = max(p(losers(i), :), lu(1, :));
        p(losers(i), :) = min(p(losers(i), :), lu(2, :));
    end    
    
    % fitness evaluation
    fitness(losers,:) = FitnessFcn(p(losers,:));
    bestever = min(bestever, min(fitness));
     
    fprintf('(%d/%d) Best fitness: %e\n', FES,maxfe,bestever);
    FES = FES + ceil(population/2);
    F(:, gen+1) = [FES; fitness];
    gen = gen + 1;
    
    if print_process && mod(FES,10000) == 0
        [~, ind] = min(fitness);
        x = p(ind, :);
        SaveProcessFcn(x, FES);
    end
end
[~, ind] = min(fitness);
x = p(ind, :);

end
