% --- Load data ---
file_path = 'C:\Users\woutg\Documents\Universiteit\Bachelor 3\Bachelorproef\Synthetic_Immune_Data_Memory_Cytokine_Realistic.csv';
opts = detectImportOptions(file_path, 'Delimiter', ';');
data = readtable(file_path, opts);

% --- Extract data ---
time = data.Time_days_;

tconv_normal = data.Tconv_Normal;
treg_normal  = data.Treg_Normal;
mreg_normal  = data.Mreg_Normal;
il2_normal   = data.IL2_Normal;

tconv_cancer = data.Tconv_Cancer;
treg_cancer  = data.Treg_Cancer;
mreg_cancer  = data.Mreg_Cancer;
il2_cancer   = data.IL2_Cancer;

tconv_auto = data.Tconv_Autoimmune;
treg_auto  = data.Treg_Autoimmune;
mreg_auto  = data.Mreg_Autoimmune;
il2_auto   = data.IL2_Autoimmune;

% --- Define color codes (easy to modify hex codes here) ---
color_normal   = '#FFA500';  % Orange
color_cancer   = '#FF0000';  % Red
color_auto     = '#0000FF';  % Blue

% Convert hex to RGB (MATLAB expects normalized RGB [0–1])
color_normal = sscanf(color_normal(2:end),'%2x%2x%2x',[1 3])/255;
color_cancer = sscanf(color_cancer(2:end),'%2x%2x%2x',[1 3])/255;
color_auto   = sscanf(color_auto(2:end),'%2x%2x%2x',[1 3])/255;

% --- Plot all in one figure ---
figure;
sgtitle('Comparison of Immune Population Dynamics: Normal vs. Cancer vs. Autoimmune');

% Subplot 1: Tconv
subplot(2,2,1);
hold on;
plot(time, tconv_normal, '--o', 'Color', color_normal, 'DisplayName', 'Normal', 'MarkerSize', 4);
plot(time, tconv_cancer, '-x',  'Color', color_cancer, 'DisplayName', 'Cancer', 'MarkerSize', 4);
plot(time, tconv_auto,   '-s',  'Color', color_auto,   'DisplayName', 'Autoimmune', 'MarkerSize', 4);
title('Tconv Dynamics');
xlabel('Time (Days)');
ylabel('Population Level (Units/L)');
legend('show');
grid on;

% Subplot 2: Treg
subplot(2,2,2);
hold on;
plot(time, treg_normal, '--o', 'Color', color_normal, 'DisplayName', 'Normal', 'MarkerSize', 4);
plot(time, treg_cancer, '-x',  'Color', color_cancer, 'DisplayName', 'Cancer', 'MarkerSize', 4);
plot(time, treg_auto,   '-s',  'Color', color_auto,   'DisplayName', 'Autoimmune', 'MarkerSize', 4);
title('Treg Dynamics');
xlabel('Time (Days)');
ylabel('Population Level (Units/L)');
legend('show');
grid on;

% Subplot 3: IL-2
subplot(2,2,3);
hold on;
plot(time, il2_normal, '--o', 'Color', color_normal, 'DisplayName', 'Normal', 'MarkerSize', 4);
plot(time, il2_cancer, '-x',  'Color', color_cancer, 'DisplayName', 'Cancer', 'MarkerSize', 4);
plot(time, il2_auto,   '-s',  'Color', color_auto,   'DisplayName', 'Autoimmune', 'MarkerSize', 4);
title('IL-2 Dynamics');
xlabel('Time (Days)');
ylabel('Population Level (Units/L)');
legend('show');
grid on;

% Subplot 4: Tmreg
subplot(2,2,4);
hold on;
plot(time, mreg_normal, '--o', 'Color', color_normal, 'DisplayName', 'Normal', 'MarkerSize', 4);
plot(time, mreg_cancer, '-x',  'Color', color_cancer, 'DisplayName', 'Cancer', 'MarkerSize', 4);
plot(time, mreg_auto,   '-s',  'Color', color_auto,   'DisplayName', 'Autoimmune', 'MarkerSize', 4);
title('Tmreg Dynamics');
xlabel('Time (Days)');
ylabel('Population Level (Units/L)');
legend('show');
grid on;