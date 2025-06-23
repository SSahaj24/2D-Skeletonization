%% Samik Banerjee June 23, 2025
% Input: The path of the skeleton output by the Skeletonisation Algorithm (path)
% Output1: An output path which gives the connected components of each section of the skeletal masks (outCC)
% Output2: The optimised output geoJSONs that can be used on the web or sent to Summarisation (outJSON)

path = '/nfs/data/main/M32/Process_Detection/Skeleton/PMD1211/corrected_masks5/';
direc = dir(fullfile(path, "*.jp2"));
outCC = '/nfs/data/main/M32/Process_Detection/Skeleton/PMD1211/CC1/';
outJSON = '/nfs/data/main/M32/Process_Detection/Skeleton/PMD1211/skelJSON1/';

for i = 1 : length(direc)
    disp("Image" + num2str(i) + " / " + num2str(length(direc)));
    img = imread(fullfile(path, direc(i).name));

    dmBW = img; %imbinarize(rgb2gray(img));
    dmBW1 = bwskel(dmBW);
    CC = graph_analysis(dmBW1);
    save(fullfile(outCC, ...
        direc(i).name(1:end-3)+ "mat"), 'CC');

    dataOut = CC2JSON(CC);

    J = jsonencode(dataOut);
    fileName = fullfile(outJSON, ...
        direc(i).name(1:end-3)+ "json");
    fid = fopen(fileName, "w");
    fwrite(fid,J);
    fclose(fid);

end
