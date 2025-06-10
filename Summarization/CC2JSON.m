
function dataOut = CC2JSON(CC)

% dataOut.features(i).geometry.coordinates = ''
dataOut.type='FeatureCollection';

for i = 1 : CC.NumObjects
    % disp(i)
    dataOut.features(i).type = 'Feature';
    dataOut.features(i).ID = i;
    dataOut.features(i).properties.stroke_width = 1;

    linePath = CC.arcProperties{1,i};
    if length(linePath)==1 
        if linePath.length
            coords = zeros(size(linePath.Pos));
            dataOut.features(i).geometry.type = 'LineString';
            % line(linePath.y, linePath.x, "Color", 'y');
            coords(:,1)=linePath.Pos(:,2);
            coords(:,2)=linePath.Pos(:,1)*-1;
            dataOut.features(i).geometry.coordinates = coords;
            dataOut.features(i).length = linePath.length;
        end
    else
        dataOut.features(i).geometry.type = 'MultiLineString';
        % dataOut.features(i).length = linePath
        lengthN = 0;
        for idx = 1 : length(linePath)
            coords = zeros(size(linePath(idx).Pos));
            
            % line(linePath(idx).y,linePath(idx).x,"Color",'y');
            coords(:,1)=linePath(idx).Pos(:,2);
            coords(:,2)=linePath(idx).Pos(:,1)*-1;
            dataOut.features(i).geometry.coordinates{idx,1} = coords;
            lengthN = lengthN + linePath(idx).length;
        end
        dataOut.features(i).length = lengthN;
    end
end

end
%%
% J = jsonencode(dataOut);
% fileName = 'out.json'; 
% fid = fopen(fileName, "w");
% fwrite(fid,J);
% fclose(fid);