function plot_qes_plume_public(windsFile, plumeFile, plotLayer, drawTerrain)
%PLOT_QES_PLUME_PUBLIC Plot a QES-Plume concentration field.
%
% This public post-processing utility is intentionally site-independent.
% It does not read or plot building geometry, monitoring-point data,
% site-specific coordinates, or any nuclear-facility input data.
%
% Usage:
%   plot_qes_plume_public(windsFile, plumeFile, plotLayer)
%   plot_qes_plume_public(windsFile, plumeFile, plotLayer, drawTerrain)
%
% Inputs:
%   windsFile    - QES-Winds NetCDF output, e.g. "..._windsOut.nc"
%   plumeFile    - QES-Plume NetCDF output, e.g. "..._plumeOut.nc"
%   plotLayer    - Vertical concentration-layer index to plot
%   drawTerrain  - true/false; optionally draw terrain contours
%
% Notes:
%   1. The script plots the last available plume-output time.
%   2. Non-positive or non-finite concentration values are omitted before
%      logarithmic conversion.
%   3. Concentration normalization, if required for a specific study, should
%      be applied explicitly before the logarithm using publicly documented
%      source information.
%
% Requirement:
%   readNetCDF.m must be available on the MATLAB path.

    if nargin < 4
        drawTerrain = true;
    end

    if ~isfile(windsFile)
        error('QES-Winds output file not found: %s', windsFile);
    end

    if ~isfile(plumeFile)
        error('QES-Plume output file not found: %s', plumeFile);
    end

    if exist('readNetCDF', 'file') ~= 2
        error('readNetCDF.m was not found on the MATLAB path.');
    end

    %% Read QES outputs
    data = struct();
    varnames = struct(); %#ok<NASGU>

    [data.winds, varnames.winds] = readNetCDF(windsFile);
    [data.plume, varnames.plume] = readNetCDF(plumeFile);

    %% Coordinates
    xPlume = data.plume.x;
    yPlume = data.plume.y;
    zPlume = data.plume.z;

    %% Extract the final plume-output time
    plumeConc = data.plume.conc;

    if ndims(plumeConc) == 4
        conc = permute(plumeConc(:, :, :, end), [2, 1, 3]);
    elseif ndims(plumeConc) == 3
        conc = permute(plumeConc, [2, 1, 3]);
    else
        error('Unexpected concentration-array dimensions: %d', ndims(plumeConc));
    end

    if plotLayer < 1 || plotLayer > size(conc, 3)
        error( ...
            'plotLayer=%d is outside the available range 1-%d.', ...
            plotLayer, size(conc, 3));
    end

    layerConc = double(conc(:, :, plotLayer));

    % Remove invalid/non-positive values before logarithmic conversion.
    layerConc(~isfinite(layerConc) | layerConc <= 0) = NaN;
    logConc = log(layerConc);

    if ~any(isfinite(logConc(:)))
        error( ...
            ['No positive concentration values are available in layer %d. ' ...
             'Check the selected layer and the Plume output/averaging settings.'], ...
            plotLayer);
    end

    fprintf('Plot layer: %d\n', plotLayer);
    if plotLayer <= numel(zPlume)
        fprintf('Layer coordinate/height: %.6g\n', zPlume(plotLayer));
    end
    fprintf('Positive concentration cells: %d\n', nnz(isfinite(logConc)));

    %% Figure
    figure('Color', 'w', 'Position', [100, 100, 800, 760]);
    ax = axes;
    hold(ax, 'on');

    %% Optional terrain contours
    if drawTerrain && isfield(data.winds, 'terrain') ...
            && isfield(data.winds, 'x') && isfield(data.winds, 'y')

        terrain = double(data.winds.terrain');
        xWind = data.winds.x;
        yWind = data.winds.y;

        finiteTerrain = terrain(isfinite(terrain));

        if ~isempty(finiteTerrain)
            tMin = min(finiteTerrain);
            tMax = max(finiteTerrain);

            if tMax > tMin
                terrainLevels = linspace(tMin, tMax, 8);
                contour( ...
                    ax, xWind, yWind, terrain, terrainLevels, ...
                    'LineColor', 'k', ...
                    'LineWidth', 0.55);
            end
        end
    end

    %% Plume
    hPlume = pcolor(ax, xPlume, yPlume, logConc);
    set(hPlume, 'EdgeColor', 'none');
    shading(ax, 'interp');

    %% Discrete plume colormap
    mymap_hex = [
        "#0243A4"
        "#004EE0"
        "#1883FF"
        "#51C87C"
        "#84D440"
        "#E8F25D"
        "#F5DE50"
        "#FEE090"
        "#FDAE61"
        "#F46D43"
        "#D73027"
        "#A50026"
    ];

    mymap = zeros(size(mymap_hex, 1), 3);

    for i = 1:size(mymap_hex, 1)
        hexColor = char(mymap_hex(i));
        mymap(i, :) = sscanf(hexColor(2:end), '%2x%2x%2x', [1, 3]) / 255;
    end

    colormap(ax, mymap);

    %% Color limits from available data
    finiteLogConc = logConc(isfinite(logConc));
    cMin = floor(min(finiteLogConc));
    cMax = ceil(max(finiteLogConc));

    if cMin == cMax
        cMin = cMin - 1;
        cMax = cMax + 1;
    end

    caxis(ax, [cMin, cMax]);

    %% Axes
    axis(ax, 'equal');
    axis(ax, 'tight');

    xlabel(ax, 'X (m)', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 12, ...
        'FontWeight', 'bold');

    ylabel(ax, 'Y (m)', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 12, ...
        'FontWeight', 'bold');

    set(ax, ...
        'FontName', 'Times New Roman', ...
        'FontSize', 12, ...
        'FontWeight', 'bold', ...
        'LineWidth', 0.8, ...
        'Box', 'on', ...
        'Layer', 'top', ...
        'TickDir', 'in');

    %% Colorbar
    cb = colorbar(ax);
    cb.Title.String = 'ln(C)';
    cb.Title.FontName = 'Times New Roman';
    cb.Title.FontSize = 12;
    cb.Title.FontWeight = 'bold';

    hold(ax, 'off');
end
