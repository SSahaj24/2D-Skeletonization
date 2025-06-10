function CC = graph_analysis(dmBW1)

%% find CC in dmBW
dmBW1 = bwareaopen(dmBW1, 5);
CC = bwconncomp(dmBW1); %// Find connected components.
L = labelmatrix(CC);
for objectidx = 1:CC.NumObjects
    if ~(mod(objectidx,100))
        disp("Component " + num2str(objectidx) + " / " + num2str(CC.NumObjects));
    end
    maskedimage = (L == objectidx);  %requires R2016b or later. earlier versions: maskedimage = bsxfun(@times, A, L == objectidx);
    CC.Node_graph{objectidx}= binaryImageGraph(maskedimage);

    CC.arcProperties{objectidx} = getArcsProperties_Skel( ...
        CC.Node_graph{objectidx}, maskedimage);

end
end
%% Show images and skeletons
% figure;
% imshow(img); hold on;
% ax1 = gca;
% for i = 1 : CC.NumObjects
%     linePath = CC.arcProperties{1,i};
%     if length(linePath)==1
%         line(linePath.y, linePath.x, "Color", 'y');
%     else
%         for idx = 1 : length(linePath)
%             line(linePath(idx).y, linePath(idx).x,"Color",'y');
%         end
%     end
% end
