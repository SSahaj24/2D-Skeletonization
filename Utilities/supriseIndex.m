%% Suprise Index
% Samik Banerjee, May 4, 2025
% Inputs: Individual Neuron densities, Tracer Density, Combined Neuron density
% Outputs: Suprise Index of each Neuron


load('Densities\100u\lh_density_SWC.mat'); % Individual Neuron Density, fileIndex
load('Densities/100u/density_PMD.mat'); % Tracer Density
load('Densities\100u\density_SWC.mat'); % SWC Density

%% Extrapolate PMD densities

% Find all the non-zero Voxels of Tracer
X = [];
for x = 1 : size(density_PMD_R,1)
    for y = 1 : size(density_PMD_R,2)
        for z = 1 : size(density_PMD_R,3)
            if (density_PMD_R(x,y,z))
                X = [X; x y z];
            end
        end
    end
end

% Finad all zero voxels for non-zeros voxels of Neurons
Y = [];
for x = 1 : size(density_SWC_R,1)
    for y = 1 : size(density_SWC_R,2)
        for z = 1 : size(density_SWC_R,3)
            if (density_SWC_R(x,y,z) && density_PMD_R(x,y,z)==0)
                Y = [Y; x y z];
            end
        end
    end
end

[W, Idx] = wiNN_3D(X,Y, 5); % Calculte wiNN

% Extrapolate the Zero Voxels of Tracer for non-xero voxels of neurons
density_PMD_R1 = density_PMD_R;
% kIdx = size(Idx,2);
for i = 1 : length(Y)
    val = 0;
    for kIdx = 1 : size(Idx,2)
        val = val + ...
            density_PMD_R(X(Idx(i,kIdx),1), ...
            X(Idx(i,kIdx),2), ...
            X(Idx(i,kIdx),3)) ...
            * W(i,kIdx);
    end
    density_PMD_R1(Y(i,1),Y(i,2),Y(i,3)) = val;
end

% Initialize for Verification of Data
vox_PMD_SWC=zeros(size(density_SWC_filewise)); 
vox_PMD=zeros(size(density_SWC_filewise)); 
vox_SWC=zeros(size(density_SWC_filewise));  
vox_PMD_gt_SWC = zeros(size(density_SWC_filewise));
S = zeros(size(density_SWC_filewise));

%% Calculate Surprise Indices for each neuron
cnt = 0;
density_PMD_N = density_PMD_R1/(sum(sum(sum(density_PMD_R1))));
for i = 1 : length(density_SWC_filewise) % for each nuron, i
    density_SWC_f_R = density_SWC_filewise{1,i}; % get individual neuron density
    density_SWC_f_N = density_SWC_f_R/(sum(sum(sum(density_SWC_f_R)))); % sum of Q_vi = 1
    sumS = 0;
    % for each voxel (v)
    for x = 1 : size(density_SWC_f_N,1)
        for y = 1 : size(density_SWC_f_N,2)
            for z = 1 : size(density_SWC_f_N,3)
                if (density_PMD_N(x,y,z) && density_SWC_f_N(x,y,z))
                    cnt = cnt +1;
                    sumS = sumS + (density_SWC_f_N(x,y,z) * ...
                        log2(density_SWC_f_N(x,y,z)/density_PMD_N(x,y,z)));
                        % sum of Q_vi * log2(Q_vi/P_v); v = each voxel (x,y,z), i = each neuron
                    if(density_SWC_f_N(x,y,z)>0)
                        vox_SWC(i) = vox_SWC(i) + 1;
                    end
                    if(density_PMD_N(x,y,z)>0)
                        vox_PMD(i) = vox_PMD(i) + 1;
                    end
                    if(density_SWC_f_N(x,y,z)>0 && density_PMD_N(x,y,z)>0)
                        vox_PMD_SWC(i) = vox_PMD_SWC(i) + 1;
                        if(density_SWC_f_N(x,y,z)<=density_PMD_N(x,y,z))
                            vox_PMD_gt_SWC(i) = vox_PMD_gt_SWC(i) + 1;
                        end
                    end
                end
            end
        end
    end
    S(i) = sumS; % Surprise index for each neuron
end

%% Sorting
[sortedS, idxS] = sort(S);
T = table(S',fileIndex');

%% Plots
[h, edges, bins] = histcounts(S, 14);
figure('Color', 'black');
hold on;

b = bar(edges(2:end), h);
xtips = b.XEndPoints;
ytips = b.YEndPoints;
labels = string(b.YData); % Convert YData to strings
text(xtips, ytips, labels, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'Color', 'g', 'FontWeight', 'bold', 'FontSize', 16);


set(gca,'Color','k')
set(gca, 'XColor', 'w', 'FontWeight', 'bold', 'FontSize', 12);
set(gca, 'YColor', 'w', 'FontWeight', 'bold', 'FontSize', 12);
set(gca, 'LineWidth', 2);

xlabel('Surprise Index', 'Color', 'y', ...
    'FontWeight', 'bold', 'FontSize', 16);
ylabel('Counts', 'Color', 'y', ...
    'FontWeight', 'bold', 'FontSize', 16);
grid on;

%% Scatter for voxSWC, Suprise Index

figure('Color', 'black');
hold on;

scatter(S, log2(vox_SWC), 'r*');


set(gca,'Color','k')
set(gca, 'XColor', 'w', 'FontWeight', 'bold', 'FontSize', 12);
set(gca, 'YColor', 'w', 'FontWeight', 'bold', 'FontSize', 12);
set(gca, 'LineWidth', 2);

xlabel('Surprise Index', 'Color', 'y', ...
    'FontWeight', 'bold', 'FontSize', 16);
ylabel('Log_2(#voxels per Neurons)', 'Color', 'y', ...
    'FontWeight', 'bold', 'FontSize', 16);
grid on;

