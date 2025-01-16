function h = shade_psth(x,y,ysd,color)
%SHADE_PSTH Summary of this function goes here
%   Detailed explanation goes here
patch([x fliplr(x)], [(y+ysd) fliplr(y-ysd)], color, 'FaceAlpha',0.2, 'EdgeColor','none');
hold on;
h = plot(x,y,"Color",color,'LineWidth',2);
end

