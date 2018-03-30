%% This is a demo code for the BLUFR performance report given result 
% files of the comparing algorithms. For your own plot, edit the
% "resultFiles" and "methods" variables.

close all; clear; clc;

resultFiles = {
    'result_sphereface_pca.mat';
    'HighDimLBP+ITML.mat'
    };

methods = {'result_sphereface_pca';'HighDimLBP+ITML'};

lineStyle = {'r', 'g', 'b', 'c', 'm', 'k', 'r--', 'g--', 'b--', 'k--'};

logFile = 'results.txt';

reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting

numMethods = length(methods);
reportVR = zeros(numMethods, 1);
reportDIR = zeros(numMethods, 1);

veriText = sprintf('Verification rates (%%) @FAR = %g%%:\n\n', reportVeriFar * 100);
osiText = sprintf('Detection and identification rates (%%) @FAR = %g%%, Rank = %d:\n\n', reportOsiFar * 100, reportRank);

%% Load results and plot curves
for i = 1 : numMethods
    s = load(resultFiles{i});
    
    % check the report point
    if ~s.reportVeriFar == reportVeriFar || ~s.reportOsiFar == reportOsiFar || ~s.reportRank
        error('Not a compatible result file.');
    end
    
    veriText = sprintf('%s%s: \t %.2f%%\n', veriText, methods{i}, s.reportVR);
    osiText = sprintf('%s%s: \t %.2f%%\n', osiText, methods{i}, s.reportDIR);

    % Plot the face verification ROC curve.
    figure(1);
    semilogx(s.meanVeriFAR * 100, s.fusedVR, lineStyle{i}, 'LineWidth', 2);
    hold on;

    % Plot the open-set face identification ROC curve at the report rank.
    figure(2);
    semilogx(s.meanOsiFAR * 100, s.fusedDIR(s.rankIndex, :), lineStyle{i}, 'LineWidth', 2);
    hold on;

    % Plot the open-set face identification CMC curve at the report FAR.
    figure(3);
    semilogx(s.rankPoints, s.fusedDIR(:, s.osiFarIndex), lineStyle{i}, 'LineWidth', 2);
    hold on;
end

%% Display the benchmark performance and output to the log file.
fprintf('%s\n%s', veriText, osiText);
fout = fopen(logFile, 'wt');
fprintf(fout, '%s\n%s', veriText, osiText);
fclose(fout);

figure(1);
set(gca, 'FontSize', 12, 'FontWeight', 'Bold');
legend(methods, 'Location', 'NorthWest');
ylim([0,100]); grid on;
xlabel('False Accept Rate (%)');
ylabel('Verification Rate (%)');
title('Face Verification ROC Curve');

figure(2);
set(gca, 'FontSize', 12, 'FontWeight', 'Bold');
legend(methods, 'Location', 'NorthWest');
grid on;
xlabel('False Accept Rate (%)');
ylabel('Detection and Identification Rate (%)');
title(sprintf('Open-set Identification ROC Curve at Rank %d', reportRank));

figure(3);
set(gca, 'FontSize', 12, 'FontWeight', 'Bold');
legend(methods);
grid on;
xlabel('Rank');
ylabel('Detection and Identification Rate (%)');
title( sprintf('Open-set Identification CMC Curve at FAR = %g%%', reportOsiFar*100) );
