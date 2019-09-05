function [f_ybest, ybest, FEs] = ccpso2(ybest, f_ybest, func_num, Max_FEs)
NP = 30; 
D = 1000;
FEs = 0;
if (func_num == 1 | func_num == 2 | func_num == 3)
  lb = -100;
  ub = 100;
end
if (func_num == 4)
  lb = -5;
  ub = 5;
end
if (func_num == 5)
  lb = -600;
  ub = 600;
end
if func_num == 6
  lb = -32;
  ub = 32;
end
if (func_num == 7)
  lb = -1;
  ub = 1;
end
x = repmat(ybest, NP, 1);
x = x + randn(NP,D) * (ub-lb)/1000;
x = bo(x, lb, ub, D);
% x = lb+rand(NP,D)*(ub-lb);
y = x;
perm_dim = randperm(D);
if D == 10
    S = [2,5];
    kk = 2;
elseif D == 100
    S = [2,5,10,50,100];
    kk = 5;
else
    S = [2,5,10,50,100,250];
    kk = 6;
end

s = ceil(rand*kk);
s = S(s);
K = D/s;
Iteration = 0;
while (FEs < Max_FEs)
    Iteration = Iteration+1;
    f_py = zeros(K,NP);
    work = 0;
    for j = 1:K
        le = s*(j-1)+1;
        ri = j*s;
        for i = 1:NP
            tmp_x = ybest;
            tmp_x(perm_dim(le:ri)) = x(i,perm_dim(le:ri));
            B = benchmark_func(tmp_x,func_num);
            FEs = FEs+1;

            if sum(abs(x(i,perm_dim(le:ri))-y(i,perm_dim(le:ri)))) > 0
                tmp_x(perm_dim(le:ri)) = y(i,perm_dim(le:ri));
                C = benchmark_func(tmp_x,func_num);
                FEs = FEs+1;
            else
                C = B;
            end

            if B < C
                y(i,perm_dim(le:ri)) = x(i,perm_dim(le:ri));
                C = B;
            end
            if C < f_ybest
                ybest(perm_dim(le:ri)) = y(i,perm_dim(le:ri));
                f_ybest = C;
                work = 1;
            end
            f_py(j,i) = C;
        end

        %**************************************************
        py_pi = zeros(NP,D);
        i = 1;
        if f_py(j,NP)<f_py(j,i) && f_py(j,NP)<f_py(j,i+1)
            py_pi(i,perm_dim(le:ri)) = x(NP,perm_dim(le:ri));
        elseif f_py(j,i)<f_py(j,NP) && f_py(j,i)<f_py(j,i+1)
            py_pi(i,perm_dim(le:ri)) = x(i,perm_dim(le:ri));
        else
            py_pi(i,perm_dim(le:ri)) = x(i+1,perm_dim(le:ri));
        end
        i = NP;
        if f_py(j,i-1)<f_py(j,i) && f_py(j,i-1)<f_py(j,1)
            py_pi(i,perm_dim(le:ri)) = x(i-1,perm_dim(le:ri));
        elseif f_py(j,i)<f_py(j,i-1) && f_py(j,i)<f_py(j,1)
            py_pi(i,perm_dim(le:ri)) = x(i,perm_dim(le:ri));
        else
            py_pi(i,perm_dim(le:ri)) = x(1,perm_dim(le:ri));
        end
        for i = 2:NP-1
            if f_py(j,i-1)<f_py(j,i) && f_py(j,i-1)<f_py(j,i+1)
                py_pi(i,perm_dim(le:ri)) = x(i-1,perm_dim(le:ri));
            elseif f_py(j,i)<f_py(j,i-1) && f_py(j,i)<f_py(j,i+1)
                py_pi(i,perm_dim(le:ri)) = x(i,perm_dim(le:ri));
            else
                py_pi(i,perm_dim(le:ri)) = x(i+1,perm_dim(le:ri));
            end
        end
       %****************************************************
       tmp = x;
        for i = 1:NP
            for jj = le:ri
                if rand <= 0.5
                    x(i,perm_dim(jj)) = y(i,perm_dim(jj))+(tan(pi*(rand-0.5)))*abs(y(i,perm_dim(jj))-py_pi(i,perm_dim(jj)));
                else
                    x(i,perm_dim(jj)) = py_pi(i,perm_dim(jj))+normrnd(0,1)*abs(y(i,perm_dim(jj))-py_pi(i,perm_dim(jj)));
                end
            end
        end
        %*************±ß½ç´¦Àí***********************
         for i = 1:NP
             for j = 1:D
                 if x(i,j) > ub | x(i,j) < lb
                     x(i,j) = tmp(i,j);
                 end
             end
         end
         %********************************************

        disp(strcat('fbest=',num2str(min(f_ybest)),' ,','FEs=',num2str(FEs)));
    end
    if work == 0
        s = ceil(rand*kk);
        s = S(s);
        K = D/s;
        perm_dim = randperm(D);
    end   


end

func_num
f_ybest
