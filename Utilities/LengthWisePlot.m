%% Top Regionwise Line Length in RAF (LH)
% Samik Banerjee, May 4, 2025
% Input: All linelengths, region wise, per category
% Output : Sorted top LineLength in each region, per category.
%          Top common regions with highest linelegths


load('Densities\LineLengths_all.mat');

T_allen = readtable('mouselist.csv'); % Mouse regionIDS of Allen
T_hash = readtable('BAP2Allen_segids_hash.csv'); % Lookup table of corresponding RAF and Allen IDs

%% Only in LH (Left Hemisphere)
% For Individual Neurons Combined
SWC_lineLength_lh = SWC_lineLength1(1:670,:); % Top 670 indices are in LH
SWC_regionWise_lh = SWC_regionWise(1:670);
SWC_regionWise_lh_norm = SWC_regionWise_lh/SWC_total;

% For STP
% Optical section Thickness=2um, Tracer Thickness=1um, Scetion Spacing=50um
% Interpolation Factor: 50/(2+1)
interpF = 16; 
STP_lineLength_lh = STP_lineLength1(1:670,:);
STP_regionWise_lh = STP_regionWise(1:670);
STP_regionWise_lh_interp = STP_regionWise_lh * interpF;
STP_regionWise_lh_norm = STP_regionWise_lh/STP_total;

% For Tracer
% Optical section Thickness=1.5um, Tracer Thickness=1um, Scetion Spacing=40um
% Interpolation Factor: 40/(1.5+1)
interpF = 16;
PMD_lineLength_lh = PMD_lineLength1(1:670,:);
PMD_regionWise_lh = PMD_regionWise(1:670);
PMD_regionWise_lh_interp = PMD_regionWise_lh * interpF;
PMD_regionWise_lh_norm = PMD_regionWise_lh/PMD_total;

%% Sorting 
top_number = 100; % to get considerable overlap in the variants
PL_indices = [171 195 304 363 84 132 972]; % RegionIDs of PL regions since the injections are in PL
%% Find all top 100 regions with highest length
%% STP
[values, indices] = sort(STP_regionWise_lh_interp, 'descend');
STP_top_values = values(1:top_number);
STP_top_indices_BAP = indices(1:top_number);

for i = 1 : length(STP_top_indices_BAP)
    STP_top_indices_allen(i) = T_hash.AllenIDs( ...
        find(T_hash.BAPIDs==STP_top_indices_BAP(i)));
    
    if STP_top_indices_allen(i)==997
        STP_top_indices_allenName(i) = {'WHOLE'}; % RegionID not present in AllenList
    else
        STP_top_indices_allenName(i) = T_allen.RegionAcronym(...
            find(T_allen.RegionID==STP_top_indices_allen(i)));
    end
end

%% Tracer
[values, indices] = sort(PMD_regionWise_lh_interp, 'descend');
PMD_top_values = values(1:top_number);
PMD_top_indices_BAP = indices(1:top_number);

for i = 1 : length(PMD_top_indices_BAP)
    PMD_top_indices_allen(i) = T_hash.AllenIDs( ...
        find(T_hash.BAPIDs==PMD_top_indices_BAP(i)));
    
    if PMD_top_indices_allen(i)==997
        PMD_top_indices_allenName(i) = {'WHOLE'};
    else
        PMD_top_indices_allenName(i) = T_allen.RegionAcronym(...
            find(T_allen.RegionID==PMD_top_indices_allen(i)));
    end
end

%% SWC
[values, indices] = sort(SWC_regionWise_lh, 'descend');
SWC_top_values = values(1:top_number);
SWC_top_indices_BAP = indices(1:top_number);

for i = 1 : length(SWC_top_indices_BAP)
    SWC_top_indices_allen(i) = T_hash.AllenIDs( ...
        find(T_hash.BAPIDs==SWC_top_indices_BAP(i)));
    
    if SWC_top_indices_allen(i)==997
        SWC_top_indices_allenName(i) = {'WHOLE'};
    else
        SWC_top_indices_allenName(i) = T_allen.RegionAcronym(...
            find(T_allen.RegionID==SWC_top_indices_allen(i)));
    end
end

%% Common: Find teh common region between top 100 regions
common_SWC_STP = intersect(SWC_top_indices_allen, STP_top_indices_allen);
common_SWC_STP_PMD = intersect(common_SWC_STP, PMD_top_indices_allen);
common_SWC_STP_PMD_noPL = setdiff(common_SWC_STP_PMD, PL_indices);

for i = 1 : length(common_SWC_STP_PMD_noPL)
    top_indices_allenName(i) = T_allen.RegionAcronym(...
        find(T_allen.RegionID==common_SWC_STP_PMD_noPL(i)));
    top_indices_densityValue_STP(i) = STP_top_values(...
        find(STP_top_indices_allen == common_SWC_STP_PMD_noPL(i)));
    top_indices_densityValue_PMD(i) = PMD_top_values(...
        find(PMD_top_indices_allen == common_SWC_STP_PMD_noPL(i)));
    top_indices_densityValue_SWC(i) = SWC_top_values(...
        find(SWC_top_indices_allen == common_SWC_STP_PMD_noPL(i)));
end

%% Sort: Top region based on Tracer Data
[top_indices_densityValue_PMD_sorted, idx] = sort(...
    top_indices_densityValue_PMD, 'descend');
for i = 1 : length(idx)
    top_indices_densityValue_STP_sorted(i) = ...
        top_indices_densityValue_STP(idx(i));
    top_indices_densityValue_SWC_sorted(i) = ...
        top_indices_densityValue_SWC(idx(i));
    top_indices_allenName_sorted{i} = ...
        top_indices_allenName{idx(i)};
end

%% Normal Plot
idxS = 1; 
idxL = 12; % No of. top regions
bc = [0.07 0.62 1.00];
figure('Color', 'black');
hold on;

top_indices_allenName_sorted_sub = top_indices_allenName_sorted(1, idxS:idxL);
top_indices_densityValue_SWC_sorted_sub = top_indices_densityValue_SWC_sorted(idxS: idxL);
top_indices_densityValue_PMD_sorted_sub = top_indices_densityValue_PMD_sorted(idxS: idxL);
top_indices_densityValue_STP_sorted_sub = top_indices_densityValue_STP_sorted(idxS: idxL);

% plot(top_indices_densityValue_STP_sorted_sub/1000000, 'Color', bc, ...
%     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', bc);
% % plot(top_indices_densityValue_PMD_sorted_sub*1000, 'r', ...
% %     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'r');
% plot(top_indices_densityValue_SWC_sorted_sub/1000000, 'g', ...
%     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'g');

plot(log10(top_indices_densityValue_STP_sorted_sub/1000000), 'Color', bc, ...
    'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', bc);
plot(log10(top_indices_densityValue_PMD_sorted_sub/1000000), 'r', ...
    'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'r');
plot(log10(top_indices_densityValue_SWC_sorted_sub/1000000), 'g', ...
    'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'g');

% semilogx([1 : length(top_indices_densityValue_STP_sorted_sub)], ...
%     (top_indices_densityValue_STP_sorted_sub/1000000), 'Color', bc, ...
%     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', bc);
% semilogx([1 : length(top_indices_densityValue_PMD_sorted_sub)], ...
%     (top_indices_densityValue_PMD_sorted_sub/1000000), 'r', ...
%     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'r');
% semilogx([1 : length(top_indices_densityValue_SWC_sorted_sub)], ...
%     (top_indices_densityValue_SWC_sorted_sub/1000000), 'g', ...
%     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'g');


xticks(1 : length(top_indices_densityValue_PMD_sorted_sub));
xticklabels(top_indices_allenName_sorted_sub);
xtickangle(45);
set(gca,'Color','k')
set(gca, 'XColor', 'w', 'FontWeight', 'bold', 'FontSize', 18);
set(gca, 'YColor', 'w', 'FontWeight', 'bold', 'FontSize', 18);
set(gca, 'LineWidth', 2);

lgd = legend('STP','fWSI','ION');
% lgd = legend('STP','ION');
lgd.FontSize = 16;
lgd.FontWeight = 'bold';
lgd.EdgeColor = 'y';
lgd.TextColor = 'white';
lgd.Orientation = 'horizontal';
lgd.Location = 'best';

xlabel('RAF Brain Regions', 'Color', 'y', ...
    'FontWeight', 'bold', 'FontSize', 16);
ylabel('log_{10} ( Length in meters )', 'Color', 'y', ...
    'FontWeight', 'bold', 'FontSize', 16);
grid on;

%% Extra if required (clear variable and run till Sorting above and then this section)
%% For Norm linelength 
%% Sorting
top_number = 100;
PL_indices = [171 195 304 363 84 132 972];
%% STP
[values, indices] = sort(STP_regionWise_lh_norm, 'descend');
STP_top_values = values(1:top_number);
STP_top_indices_BAP = indices(1:top_number);

for i = 1 : length(STP_top_indices_BAP)
    STP_top_indices_allen(i) = T_hash.AllenIDs( ...
        find(T_hash.BAPIDs==STP_top_indices_BAP(i)));
    
    if STP_top_indices_allen(i)==997
        STP_top_indices_allenName(i) = {'WHOLE'};
    else
        STP_top_indices_allenName(i) = T_allen.RegionAcronym(...
            find(T_allen.RegionID==STP_top_indices_allen(i)));
    end
end
%% SWC
[values, indices] = sort(SWC_regionWise_lh_norm, 'descend');
SWC_top_values = values(1:top_number);
SWC_top_indices_BAP = indices(1:top_number);
for i = 1 : length(SWC_top_indices_BAP)
    SWC_top_indices_allen(i) = T_hash.AllenIDs( ...
        find(T_hash.BAPIDs==SWC_top_indices_BAP(i)));   
    if SWC_top_indices_allen(i)==997
        SWC_top_indices_allenName(i) = {'WHOLE'};
    else
        SWC_top_indices_allenName(i) = T_allen.RegionAcronym(...
            find(T_allen.RegionID==SWC_top_indices_allen(i)));
    end
end
%% common
common_SWC_STP = intersect(SWC_top_indices_allen, STP_top_indices_allen);
common_SWC_STP_PMD = intersect(common_SWC_STP, PMD_top_indices_allen);
common_SWC_STP_PMD_noPL = setdiff(common_SWC_STP, PL_indices);
for i = 1 : length(common_SWC_STP_PMD_noPL)
    top_indices_allenName(i) = T_allen.RegionAcronym(...
        find(T_allen.RegionID==common_SWC_STP_PMD_noPL(i)));
    top_indices_densityValue_STP(i) = STP_top_values(...
        find(STP_top_indices_allen == common_SWC_STP_PMD_noPL(i)));
    top_indices_densityValue_PMD(i) = PMD_top_values(...
        find(PMD_top_indices_allen == common_SWC_STP_PMD_noPL(i)));
    top_indices_densityValue_SWC(i) = SWC_top_values(...
        find(SWC_top_indices_allen == common_SWC_STP_PMD_noPL(i)));
end
%% Sort
[top_indices_densityValue_PMD_sorted, idx] = sort(...
    top_indices_densityValue_PMD, 'descend');
for i = 1 : length(idx)
    top_indices_densityValue_STP_sorted(i) = ...
        top_indices_densityValue_STP(idx(i));
    top_indices_densityValue_SWC_sorted(i) = ...
        top_indices_densityValue_SWC(idx(i));
    top_indices_allenName_sorted{i} = ...
        top_indices_allenName{idx(i)};
end
%% Normal Plot
idxS = 1;
idxL = 25;
bc = [0.07 0.62 1.00];
figure('Color', 'black');
hold on;

top_indices_allenName_sorted_sub = top_indices_allenName_sorted(1, idxS:idxL);
top_indices_densityValue_SWC_sorted_sub = top_indices_densityValue_SWC_sorted(idxS: idxL);
top_indices_densityValue_PMD_sorted_sub = top_indices_densityValue_PMD_sorted(idxS: idxL);
top_indices_densityValue_STP_sorted_sub = top_indices_densityValue_STP_sorted(idxS: idxL);

% plot(top_indices_densityValue_STP_sorted_sub, 'Color', bc, ...
%     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', bc);
% % plot(top_indices_densityValue_PMD_sorted_sub*1000, 'r', ...
% %     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'r');
% plot(top_indices_densityValue_SWC_sorted_sub, 'g', ...
%     'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'g');

plot(log(top_indices_densityValue_STP_sorted_sub), 'Color', bc, ...
    'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', bc);
plot(log(top_indices_densityValue_PMD_sorted_sub), 'r', ...
    'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'r');
plot(log(top_indices_densityValue_SWC_sorted_sub), 'g', ...
    'LineWidth', 2, 'Marker', 'diamond', 'MarkerFaceColor', 'g');

xticks(1 : length(top_indices_densityValue_SWC_sorted_sub));
xticklabels(top_indices_allenName_sorted_sub);
xtickangle(45);
set(gca,'Color','k')
set(gca, 'XColor', 'w', 'FontWeight', 'bold', 'FontSize', 18);
set(gca, 'YColor', 'w', 'FontWeight', 'bold', 'FontSize', 18);
set(gca, 'LineWidth', 2);

% lgd = legend('STP','fWSI','ION');
lgd = legend('STP','ION');
lgd.FontSize = 16;
lgd.FontWeight = 'bold';
lgd.EdgeColor = 'y';
lgd.TextColor = 'white';
lgd.Orientation = 'horizontal';
lgd.Location = 'best';

xlabel('RAF Brain Regions', 'Color', 'y', ...
    'FontWeight', 'bold', 'FontSize', 16);
ylabel('log (normalized Length)', 'Color', 'y', ...
    'FontWeight', 'bold', 'FontSize', 16);
grid on;
