h0 = figure;
h1 = imshow('F:\no-cc-cso-1\GT25_iter_1_without_smoothing.png');
h2 = uicontrol('style','text','Position',[30 15 100 15],'string','non');
set(h1,'ButtonDownFcn',@clicky);

function clicky(varargin) 
    a=get(gca,'Currentpoint');
    set(findobj('style','text'),'String',strcat('x:',num2str(a(1,1)),'y:',num2str(a(1,2))));    
end