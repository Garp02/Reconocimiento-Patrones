%% Programa de reconocimiento de patrones con LBP
clc; clear; close all;

%% 1. Cargar dataset KTH-TIPS2
imds = imageDatastore('KTH_TIPS',...
    'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');


% Preprocesamiento: Convertir a escala de grises y redimensionar
imds.ReadFcn = @(loc)imresize(rgb2gray(imread(loc)), [200 200]);

%% 2. Extracción de características LBP
numImages = numel(imds.Files);
%features = zeros(numImages, 256); % 256 bins para el histograma LBP

radius = 2;          % Radio del vecindario
numNeighbors = 8;    % Número de puntos de muestreo
numBins = 128; 

function lbp_image = computeLBP(image)
    [rows, cols] = size(image);
    lbp_image = zeros(rows-2, cols-2, 'uint8');
    
    for i = 2:rows-1
        for j = 2:cols-1
            center = image(i, j);
            binary_pattern = zeros(1, 8);
            
            % Comparar con los 8 vecinos (en orden clockwise)
            neighbors = [image(i-1, j-1), image(i-1, j), image(i-1, j+1), ...
                         image(i, j+1),   image(i+1, j+1), image(i+1, j), ...
                         image(i+1, j-1), image(i, j-1)];
            
            binary_pattern = (neighbors >= center);
            lbp_value = sum(binary_pattern .* 2.^(0:7)); % Convertir binario a decimal
            lbp_image(i-1, j-1) = lbp_value;
        end
    end
end

function features = extractLBPFeatures(images)
    num_images = numel(images.Files);
    features = zeros(num_images, 256); % 256 bins

    for k = 1:num_images
        lbp = computeLBP(readimage(images, k));
        hist = imhist(lbp, 256)';
        features(k, :) = hist / sum(hist); % Normalizar
    end

end

features = extractLBPFeatures(imds); % Matriz de características [30x256]

%% 3. Preparar datos para clasificación
labels = imds.Labels;
data = array2table(features);
data.Label = labels;

