
function [ y ] = EvalFun11(x,F_yx,B_yx,U_yx,F_rgb,B_rgb,U_rgb,F_mindist,B_mindist)%
%EVALFUN11 PAMI11评价函数
%   三个目标，分别是颜色误差，到前景点距离，到背景点距离
%     global F_yx B_yx U_yx F_rgb B_rgb U_rgb F_mindist B_mindist;

    x = reshape(x,[length(x)/2,2]);
    x(:,1) = min(length(F_yx),x(:,1));
    x(:,2) = min(length(B_yx),x(:,2));
    Fx_rgb = F_rgb(x(:,1),:);
    Bx_rgb = B_rgb(x(:,2),:);
    Fx_Bx_rgb = Fx_rgb-Bx_rgb;

%     if(sum(Fx_Bx_rgb.*Fx_Bx_rgb,2)<=0)
%         disp('sum error');
%     end
    alpha = sum((U_rgb-Bx_rgb).*(Fx_Bx_rgb),2)./(sum(Fx_Bx_rgb.*Fx_Bx_rgb,2)+1);
    alphaX3 = [alpha,alpha,alpha];
    %% color distance
    Obj1 = sum((U_rgb - (alphaX3.*Fx_rgb+(1-alphaX3).*Bx_rgb)).^2,2).^0.5;

    %% spatial distance
    Fx_yx = F_yx(x(:,1),:);
    Bx_yx = B_yx(x(:,2),:);
%     if(min(F_mindist)<=0||min(B_mindist)<=0)
%         disp('mindist error');
%     end

    Obj2 = (sum((Fx_yx-U_yx).^2,2).^0.5./F_mindist).^0.5;
    Obj3 = (sum((Bx_yx-U_yx).^2,2).^0.5./B_mindist).^0.5;
    y = sum(mean([Obj1,Obj2,Obj3]));
end

