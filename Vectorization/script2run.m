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