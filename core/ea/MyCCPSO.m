function [ x,fval ] = MyCCPSO( FitnessFcn,numberOfVariables,lb,ub,mfe, initial_x)
%MYCCPSO Summary of this function goes here
%   Detailed explanation goes here
% format short e;
%
NP = 100;
D = numberOfVariables;
Max_FEs = mfe;
w = 0.4;
W = repmat(0.4,NP,1);
c1 = 2;
c2 = 2;
limit = zeros(NP,1);
%
% global initial_flag;
% global FE;
% global func_num;

%     disp(['func_num: ', num2str(func_num)]);
%     filename = strcat('test',num2str(func_num),'.txt');
%     fid = fopen(filename,'wt');
%     if(func_num == 2 || func_num == 5 || func_num == 10 || func_num == 15)
%         lb = -5;
%         ub = 5;
%     elseif(func_num == 3 || func_num == 6 || func_num == 11 || func_num == 16)
%         lb = -32;
%         ub = 32;
%     else
%         lb = -100;
%         ub = 100;
%     end
% lu = [lb;ub];
% for run = 1:25
    stop_flag = 0;
%     disp(['run: ', num2str(run)]);
    initial_flag = 0;
    FE = 0;
    
    V = zeros(NP,D);
    X=repmat(lb,NP,1)+rand(NP,D).*repmat(ub-lb,NP,1);
    
    % 如果有初始解，把第一个设置为初始解
    if exist('initial_x', 'var') && ~isempty(initial_x)
       X(1, :) = initial_x; 
    end
    y = X;
    fitValue = FitnessFcn(X);
    FE = FE + NP;
    last_fit = fitValue;
    [fit_ybest, fit_ybest_row] = min(fitValue);
    ybest = X(fit_ybest_row,:);
    %disp(ybest);
    if D==10
        S=[2,5];
        kk=2;
    elseif D==100
        S=[2,5,10,50,100];
        kk=5;
    else
        S=[2,5,10,50];
        kk=4;
    end
    s=ceil(rand*kk);
    s=S(s);
    K=D/s;
    perm_dim=randperm(D);
%     disp(['K: ', num2str(K)]);
    
    while (FE < Max_FEs)
        r1 = rand(NP,D);
        r2 = rand(NP,D);
        ybest_mat = repmat(ybest, NP, 1);
        
        for p = 1:NP
            V = W(p)*V;
        end
        %w = 0.5 - 0.2 * FE / Max_FEs;
        V = V + c1*r1.*(y-X) + c2*r2.*(ybest_mat-X);
        X = X + V;
        
        X = min(X,repmat(ub,NP,1));
        X = max(X,repmat(lb,NP,1));
        %disp(W);
        
        for j=1:K
            %for j = 1:1
            le=s*(j-1)+1;
            ri=j*s;
            
            for i=1:NP
                %for i = 1:1
                tmp_x=ybest;
                tmp_x(perm_dim(le:ri))=X(i,perm_dim(le:ri));
                %disp(['s: ', num2str(s)]);
                %disp(X(i,perm_dim(le:ri)));
                B=FitnessFcn(tmp_x);
                FE = FE + 1;
                if(FE > Max_FEs)
                    stop_flag = 1;
                    break;
                end
                
                if X(i,perm_dim(le:ri))~=y(i,perm_dim(le:ri))
                    tmp_x(perm_dim(le:ri))=y(i,perm_dim(le:ri));
                    C=FitnessFcn(tmp_x);
                    FE = FE + 1;
                    if(FE > Max_FEs)
                        stop_flag = 1;
                        break;
                    end
                else
                    C=B;
                end
                if B<C
                    y(i,perm_dim(le:ri))=X(i,perm_dim(le:ri));
                    C=B;
                end
                if C<fit_ybest
                    ybest(perm_dim(le:ri))=y(i,perm_dim(le:ri));
                    fit_ybest=C;
                    work=1;
                end
                %disp(ybest(perm_dim(le:ri)));
            end
            %}
            if(stop_flag == 1)
                break;
            end
        end
        
        new_fit = FitnessFcn(X);
        FE = FE + NP;
        for i = 1:NP
            if(new_fit(i) >= last_fit(i))
                limit(i) = limit(i) + 1;
            end
            if(limit(i) == 3)
                temp = W(i);
                W(i) = 0.5 - 0.2 * rand;
                while(W(i)==temp)
                    W(i) = 0.5 - 0.2 * rand;
                end
            end
            
        end
        last_fit = new_fit;
        %}
%         disp(['FEs = ', num2str(FE)]);
%         disp(fit_ybest);
    end
    x = ybest;
    fval = fit_ybest;
%     fprintf(fid,'%d\n',fit_ybest);
end

% fclose(fid);



