% --- Clear environment ---
clear; close all; clc;

% --- Load Data ---
file_path = 'C:\Users\woutg\Documents\Universiteit\Bachelor 3\Bachelorproef\Synthetic_Immune_Data_Memory_Cytokine_Realistic.csv';
data = readtable(file_path, 'Delimiter', ';');

time = data.Time_days_;

% Normal condition
tconv_normal = data.Tconv_Normal;
treg_normal  = data.Treg_Normal;
mreg_normal  = data.Mreg_Normal;
il2_normal   = data.IL2_Normal;

% Cancer condition
tconv_cancer = data.Tconv_Cancer;
treg_cancer  = data.Treg_Cancer;
mreg_cancer  = data.Mreg_Cancer;
il2_cancer   = data.IL2_Cancer;

% Autoimmune condition
tconv_auto = data.Tconv_Autoimmune;
treg_auto  = data.Treg_Autoimmune;
mreg_auto  = data.Mreg_Autoimmune;
il2_auto   = data.IL2_Autoimmune;

% --- STYLE SETTINGS (easy to edit) ---
% Colors (Hex)
color_normal = '#000000';     % orange
color_cancer = '#93A6C9';     % red
color_auto   = '#006CA9';     % blue

% Line & Marker style
lw = 1.3; % line width
ms = 4.5;   % marker size

% Font sizes
font_label  = 14; % axis labels
font_title  = 20; % subplot titles
font_legend = 12; % legends
font_sgt    = 28; % super title

% --- Create Figure with Tiled Layout ---
fig = figure;
t = tiledlayout(2,2, 'TileSpacing','compact', 'Padding','compact');
set(gcf, 'Color', 'w'); % white background

% --- Tconv Dynamics ---
nexttile;
hold on;
plot(time, tconv_normal, 'Color',color_normal, 'LineWidth',lw, 'Marker','o', 'MarkerSize',ms);
plot(time, tconv_cancer, 'Color',color_cancer, 'LineWidth',lw, 'Marker','s', 'MarkerSize',ms);
plot(time, tconv_auto,   'Color',color_auto,   'LineWidth',lw, 'Marker','d', 'MarkerSize',ms);
title('Tconv Dynamics', 'FontSize', font_title);
xlabel('Time (Days)', 'FontSize', font_label);
ylabel('Population Level (Units/L)', 'FontSize', font_label);
legend('Normal','Cancer','Autoimmune','Location','best','FontSize',font_legend);
grid on;

% --- Treg Dynamics ---
nexttile;
hold on;
plot(time, treg_normal, 'Color',color_normal, 'LineWidth',lw, 'Marker','o', 'MarkerSize',ms);
plot(time, treg_cancer, 'Color',color_cancer, 'LineWidth',lw, 'Marker','s', 'MarkerSize',ms);
plot(time, treg_auto,   'Color',color_auto,   'LineWidth',lw, 'Marker','d', 'MarkerSize',ms);
title('Treg Dynamics', 'FontSize', font_title);
xlabel('Time (Days)', 'FontSize', font_label);
ylabel('Population Level (Units/L)', 'FontSize', font_label);
legend('Normal','Cancer','Autoimmune','Location','best','FontSize',font_legend);
grid on;

% --- IL-2 Dynamics ---
nexttile;
hold on;
plot(time, il2_normal, 'Color',color_normal, 'LineWidth',lw, 'Marker','o', 'MarkerSize',ms);
plot(time, il2_cancer, 'Color',color_cancer, 'LineWidth',lw, 'Marker','s', 'MarkerSize',ms);
plot(time, il2_auto,   'Color',color_auto,   'LineWidth',lw, 'Marker','d', 'MarkerSize',ms);
title('IL-2 Dynamics', 'FontSize', font_title);
xlabel('Time (Days)', 'FontSize', font_label);
ylabel('Population Level (Units/L)', 'FontSize', font_label);
legend('Normal','Cancer','Autoimmune','Location','best','FontSize',font_legend);
grid on;

% --- Mreg Dynamics ---
nexttile;
hold on;
plot(time, mreg_normal, 'Color',color_normal, 'LineWidth',lw, 'Marker','o', 'MarkerSize',ms);
plot(time, mreg_cancer, 'Color',color_cancer, 'LineWidth',lw, 'Marker','s', 'MarkerSize',ms);
plot(time, mreg_auto,   'Color',color_auto,   'LineWidth',lw, 'Marker','d', 'MarkerSize',ms);
title('Tmreg Dynamics', 'FontSize', font_title);
xlabel('Time (Days)', 'FontSize', font_label);
ylabel('Population Level (Units/L)', 'FontSize', font_label);
legend('Normal','Cancer','Autoimmune','Location','best','FontSize',font_legend);
grid on;

% --- Overall Title ---
sgtitle('Comparison of Immune Population Dynamics', 'FontSize', font_sgt, 'FontWeight','bold');

% --- Export High-Quality Images ---
exportFolder = 'C:\Users\woutg\Documents\Universiteit\Bachelor 3\Bachelorproef\figures\';
if ~exist(exportFolder, 'dir')
    mkdir(exportFolder);
end

exportgraphics(fig, fullfile(exportFolder, 'Immune_Population_Dynamics.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(exportFolder, 'Immune_Population_Dynamics.pdf'), 'ContentType', 'vector');
