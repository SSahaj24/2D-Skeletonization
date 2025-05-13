%% Line Length Region-wise and Section-wise(Individual) RAF
% Samik Banerjee, May 4, 2025
% Input: 20 micron labeled volume, Indivdual Neurons line strings/ STP
%        linestrings / Tracer line strings
% Output : LineLength in each region, per category.


load('Densities\Bubble_labels_20u.mat');
Bubble_labels_RAF = permute(Bubble_labels_RAF, [3 2 1]);

U = unique(Bubble_labels_RAF); % Unique Labesl in RAF
U1 = U(2:end); % Ignore label =0; which is not in atlas

%% SWC
direc = dir('Densities/Lines/lines_SWC/*.npy'); % I/P: Indivdual Neurons line strings
SWC_lineLength = zeros([length(U) length(direc)]);

for files = 1 : length(direc)
    disp(direc(files).name);
    l1 = load_np(fullfile(direc(files).folder, direc(files).name));
    cnt(files) = 0;
    for i = 1 : length(l1)
        % Sub-Voxel location of line Coords
        x1(1) = l1(i,1,1)/20 + 401;
        x1(2) = l1(i,2,1)/20 + 401;
        y1(1) = l1(i,1,2)/20 + 301;
        y1(2) = l1(i,2,2)/20 + 301;
        z1(1) = l1(i,1,3)/20 + 401;
        z1(2) = l1(i,2,3)/20 + 401;

        if (x1(1)<440  && x1(2)<440 && ...
                x1(1)>0  && x1(2)>0 && ...
                y1(1)<600  && y1(2)<600 && ...
                y1(1)>0  && y1(2)>0 && ...
                z1(1)<800  && z1(2)<800 && ...
                z1(1)>0  && z1(2)>0) % Check values within atlas volume 
            if (ceil(x1(1))==ceil(x1(2)) && ...
                    ceil(y1(1))==ceil(y1(2)) && ...
                    ceil(z1(1))==ceil(z1(2)))

                lbl = Bubble_labels_RAF(ceil(x1(1)), ceil(y1(1)), ceil(z1(1))); % Region label

                SWC_lineLength(find(U==lbl), files) = ...
                    SWC_lineLength(find(U==lbl), files) + ...
                    norm([x1(1),y1(1),z1(1)] - [x1(2),y1(2),z1(2)]) * 20; % length in microns

            else if (ceil(norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] - ...
                        [ceil(x1(2)),ceil(y1(2)),ceil(z1(2))]))==1)

                    lbl1 = Bubble_labels_RAF(ceil(x1(2)), ceil(y1(2)), ceil(z1(2))); % Region label

                    SWC_lineLength(find(U==lbl1), files) = ...
                        SWC_lineLength(find(U==lbl1), files) + ...
                        norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] ...
                        - [x1(2),y1(2),z1(2)]) * 20; % length in microns

                    lbl2 = Bubble_labels_RAF(ceil(x1(1)), ceil(y1(1)), ceil(z1(1))); % Region label

                    SWC_lineLength(find(U==lbl2), files) = ...
                        SWC_lineLength(find(U==lbl2), files) + ...
                        norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] ...
                        - [x1(1),y1(1),z1(1)]) * 20; % length in microns
            else
                cnt(files) = cnt(files) + 1;
            end
            end

        end
    end
end
SWC_lineLength1 = SWC_lineLength(2:end,:); % O/P: Individual Line length in each Region

%% STP
lines_STP = load_np('Densities/STP_json_RAF_space.npy'); % I/P: STP Line Strings section-wise
STP_lineLength = zeros([length(U) length(lines_STP)]);

for files = 1 : length(lines_STP)
    disp(files);
    l1 = lines_STP{1,files};
    cnt(files) = 0; cnt1(files) = 0;
    for i = 1 : length(l1)
        % Sub-Voxel location of line Coords
        x1(1) = l1(i,1,1)/20 + 401;
        x1(2) = l1(i,2,1)/20 + 401;
        y1(1) = l1(i,1,2)/20 + 301;
        y1(2) = l1(i,2,2)/20 + 301;
        z1(1) = l1(i,1,3)/20 + 401;
        z1(2) = l1(i,2,3)/20 + 401;

        if (x1(1)<440  && x1(2)<440 && ...
                x1(1)>0  && x1(2)>0 && ...
                y1(1)<600  && y1(2)<600 && ...
                y1(1)>0  && y1(2)>0 && ...
                z1(1)<800  && z1(2)<800 && ...
                z1(1)>0  && z1(2)>0)
            if (ceil(x1(1))==ceil(x1(2)) && ...
                    ceil(y1(1))==ceil(y1(2)) && ...
                    ceil(z1(1))==ceil(z1(2)))

                lbl = Bubble_labels_RAF(ceil(x1(1)), ceil(y1(1)), ceil(z1(1)));

                STP_lineLength(find(U==lbl), files) = ...
                    STP_lineLength(find(U==lbl), files) + ...
                    norm([x1(1),y1(1),z1(1)] - [x1(2),y1(2),z1(2)]) * 20;

            else if (ceil(norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] - ...
                        [ceil(x1(2)),ceil(y1(2)),ceil(z1(2))]))==1)

                    lbl1 = Bubble_labels_RAF(ceil(x1(2)), ceil(y1(2)), ceil(z1(2)));

                    STP_lineLength(find(U==lbl1), files) = ...
                        STP_lineLength(find(U==lbl1), files) + ...
                        norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] ...
                        - [x1(2),y1(2),z1(2)]) * 20;

                    lbl2 = Bubble_labels_RAF(ceil(x1(1)), ceil(y1(1)), ceil(z1(1)));

                    STP_lineLength(find(U==lbl2), files) = ...
                        STP_lineLength(find(U==lbl2), files) + ...
                        norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] ...
                        - [x1(1),y1(1),z1(1)]) * 20;
            else
                cnt(files) = cnt(files) + 1;
            end
            end
        else
            cnt1(files) = cnt1(files) + 1;
        end
    end
end
STP_lineLength1 = STP_lineLength(2:end,:); % O/P: Sectionwise Line length in each Region

%% PMD
load('Densities\Fluoro_json_RAF_space_allSections.mat'); % I/P: Tracer Line Strings section-wise
PMD_lineLength = zeros([length(U) length(lines_PMD)]);

for files = 1 : length(lines_PMD)
    disp(files);
    l1 = lines_PMD{1,files};
    cnt(files) = 0;
    for i = 1 : length(l1)
        % Sub-Voxel location of line Coords
        x1(1) = l1(i,1,1)/20 + 401;
        x1(2) = l1(i,2,1)/20 + 401;
        y1(1) = l1(i,1,2)/20 + 301;
        y1(2) = l1(i,2,2)/20 + 301;
        z1(1) = l1(i,1,3)/20 + 401;
        z1(2) = l1(i,2,3)/20 + 401;

        if (x1(1)<440  && x1(2)<440 && ...
                x1(1)>0  && x1(2)>0 && ...
                y1(1)<600  && y1(2)<600 && ...
                y1(1)>0  && y1(2)>0 && ...
                z1(1)<800  && z1(2)<800 && ...
                z1(1)>0  && z1(2)>0)
            if (ceil(x1(1))==ceil(x1(2)) && ...
                    ceil(y1(1))==ceil(y1(2)) && ...
                    ceil(z1(1))==ceil(z1(2)))

                lbl = Bubble_labels_RAF(ceil(x1(1)), ceil(y1(1)), ceil(z1(1)));

                PMD_lineLength(find(U==lbl), files) = ...
                    PMD_lineLength(find(U==lbl), files) + ...
                    norm([x1(1),y1(1),z1(1)] - [x1(2),y1(2),z1(2)]) * 20;

            else if (ceil(norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] - ...
                        [ceil(x1(2)),ceil(y1(2)),ceil(z1(2))]))==1)

                    lbl1 = Bubble_labels_RAF(ceil(x1(2)), ceil(y1(2)), ceil(z1(2)));

                    PMD_lineLength(find(U==lbl1), files) = ...
                        PMD_lineLength(find(U==lbl1), files) + ...
                        norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] ...
                        - [x1(2),y1(2),z1(2)]) * 20;

                    lbl2 = Bubble_labels_RAF(ceil(x1(1)), ceil(y1(1)), ceil(z1(1)));

                    PMD_lineLength(find(U==lbl2), files) = ...
                        PMD_lineLength(find(U==lbl2), files) + ...
                        norm([ceil(x1(1)),ceil(y1(1)),ceil(z1(1))] ...
                        - [x1(1),y1(1),z1(1)]) * 20;
            else
                cnt(files) = cnt(files) + 1;
            end
            end

        end
    end
end

PMD_lineLength1 = PMD_lineLength(2:end,:); % O/P: Sectionwise Line length in each Region

%% Total Length
PMD_total = sum(sum(PMD_lineLength1));
STP_total = sum(sum(STP_lineLength1));
SWC_total = sum(sum(SWC_lineLength1));

PMD_regionWise = sum(PMD_lineLength1, 2);
STP_regionWise = sum(STP_lineLength1, 2);
SWC_regionWise = sum(SWC_lineLength1, 2);

tempPMD = sum(PMD_lineLength1,1); % Find Missing Sections, where Length = 0

regionWiseLength = table(PMD_regionWise, STP_regionWise, SWC_regionWise, U1); 
