

% Plot quantization function
figure;
quantization_step = 16;
quantizer_values = -100:1:100;
quantizer_function = round(quantizer_values / quantization_step) * quantization_step;
plot(quantizer_values, quantizer_function, 'LineWidth', 2);
xlabel('Input Value');
ylabel('Quantized Value');
grid on;

% Generate DCT-II transform matrix A
M = 8.0; % MxM block
A = zeros(M, M);

for i = 1:M
    for k = 1:M
        if i == 1
            A(i, k) = sqrt(1/M) * cos(((2*(k-1)+1)*pi*(i-1))/(2*M));
        else
            A(i, k) = sqrt(2/M) * cos(((2*(k-1)+1)*pi*(i-1))/(2*M));
        end
    end
end

% Display the DCT matrix A
disp('DCT-II Transform Matrix A:');
disp(A);

% Test with a random 8x8 block
image_block = rand(8, 8);

% Forward DCT
dct_coefficients = A * image_block * A';

% Inverse DCT
reconstructed_image = A' * dct_coefficients * A;

% Display the results
figure;
subplot(1, 3, 1), imshow(image_block), title('Original Image Block');
subplot(1, 3, 2), imshow(log(abs(dct_coefficients) + 1), []), colormap(gca, 'jet'), title('DCT Coefficients (log scale)');
subplot(1, 3, 3), imshow(reconstructed_image), title('Reconstructed Image Block');


% Finding PSNR based on different bit rates

% Load images and apply DCT
original_boats_image=imread('boats512x512.tif');
original_boats_dct_coefficients = blockproc(original_boats_image, [8 8], @(block_struct) dct2(block_struct.data));

original_harbour_image=imread('harbour512x512.tif');
original_harbour_dct_coefficients = blockproc(double(original_harbour_image), [8 8], @(block_struct) dct2(block_struct.data));

original_peppers_image=imread('peppers512x512.tif');
original_peppers_dct_coefficients = blockproc(double(original_peppers_image), [8 8], @(block_struct) dct2(block_struct.data));
% Trying different quantizations on the images
quantization_step_array = 2.^(0:9);

psnrs = arrayfun(@(x) calculate_psnr(x, original_boats_dct_coefficients, original_harbour_dct_coefficients, original_peppers_dct_coefficients), quantization_step_array)

entropys = arrayfun(@(x) calculate_entropy(x, original_boats_dct_coefficients, original_harbour_dct_coefficients, original_peppers_dct_coefficients), quantization_step_array);

plot(entropys, psnrs)
xlabel('Pixel bit rate');
ylabel('PSNR (dB)');


Part 2: FWT-based Image Compression
Section 3.1 and 3.2
import signal.*
import wavelet.*
coeff= load('coeffs.mat');
% Load a sample 512x512 image (replace 'your_image.jpg' with the path to your image)
image = imread('harbour512x512.tif');
%image = imread('boats512x512.tif');
%image = imread('peppers512x512.tif');
% Convert to double precision
signal = double(image(:));
db8 = [-0.0001174768, 0.0006754494, -0.0003917404, -0.0048703530, 0.0087460940, 0.0139810279, -0.0440882539, -0.0173693010, 0.1287474266, 0.0004724846, -0.2840155430, -0.0158291053, 0.5853546837, 0.6756307363, 0.3128715909, 0.0544158422];

% Get the wavelet coefficient
pros_signal = analysis_filter_bank(signal, db8);

% Reconstruct the image from the coefficient
res_signal = synthesis_filter_bank(pros_signal, db8);

% Display the reconstructed and original images
res_image = reshape(res_signal, size(image));
imshow(image);
title('Original Image');
imshow(uint8(res_image));
title('Reconstructed Image');

return_coefficients = iterative_fwt(signal, db8, 4);
iterative_reconstructed_signal = iterative_inv_fwt(return_coefficients,db8,4);
iterative_reconstructed_image = reshape(iterative_reconstructed_signal, size(image));
imshow(uint8(iterative_reconstructed_image));
title("Iterative Reconstructed Image");

% Access the scale 1 coefficients
scale_one_coeffs = return_coefficients{1};
approximation_coeffs_one = scale_one_coeffs(1,:);
count_one = 0:(length(approximation_coeffs_one)-1);
detail_coeffs_one = scale_one_coeffs(2,:);


figure;
plot(count_one, approximation_coeffs_one);
title('Approximation Coefficients for Scale 1');
ylabel('Amplitude');

figure;
plot(count_one, detail_coeffs_one);
title("Detail Coefficients for Scale 1");
ylabel('Amplitude');

% Access the scale 4 coefficients from the return_coefficients
scale_four_coeffs = return_coefficients{4};
approximation_coeffs_four = scale_four_coeffs(1,:);
detail_coeffs_four = scale_four_coeffs(2,:);

count_four = (0:length(approximation_coeffs_four)-1);
figure;
plot(count_four, approximation_coeffs_four);
title('Approximation Coefficients for Scale 4');
ylabel('Amplitude');

figure;
plot(count_four, detail_coeffs_four);
title('Detail Coefficients for Scale 4');
ylabel('Amplitude');
Section 3.3 and 3.4
% Define a range of quantization step sizes (powers of 2)
quantization_steps = 2.^(0:9);

% Initialize arrays to store PSNR and MSE values for each quantization level
psnr_values = zeros(size(quantization_steps));
mse_reconstructed = zeros(size(quantization_steps));
mse_quantized_coeffs = zeros(size(quantization_steps));
psnr_quantized = zeros(1, numel(quantization_steps));
% Get the wavelet coefficients
return_coefficients = iterative_fwt(signal, db8, 4);

% Loop through different quantization step sizes
for i = 1:length(quantization_steps)
    % Apply uniform quantization to the coefficients
    step_size = quantization_steps(i);
    quantized_coeffs = uniform_quantizer(return_coefficients, step_size);

    % Reconstruct the signal from quantized coefficients
    quantized_reconstructed_signal = iterative_inv_fwt(quantized_coeffs, db8, 4);
    quantized_reconstructed_image = reshape(quantized_reconstructed_signal, size(image));

    % Calculate PSNR between original and reconstructed images after quantization
    psnr_values(i) = calculate_psnr2(double(image), quantized_reconstructed_image);

    % Calculate PSNR between original image and reconstructed image from quantized coefficients
    %psnr_quantized(i) = calculate_psnr2(double(image), reconstructed_from_quantized_image);

    % Calculate MSE between original and reconstructed images after quantization
    mse_reconstructed(i) = calculate_mse(double(image), quantized_reconstructed_image);

    % Reconstruct the signal from non-quantized coefficients for comparison
    non_quantized_reconstructed_signal = iterative_inv_fwt(return_coefficients, db8, 4);
    non_quantized_reconstructed_image = reshape(non_quantized_reconstructed_signal, size(image));

    % Calculate MSE between original and reconstructed images without quantization
    mse_quantized_coeffs(i) = calculate_mse(double(image), non_quantized_reconstructed_image);
end

% Plotting the rate-PSNR curve
figure;
plot(log2(quantization_steps), psnr_values, 'bo-');
xlabel('log2(Quantizer Step Size)');
ylabel('PSNR');
title('Rate-PSNR Curve');

disp(['PSNR of reconstructed image: ', num2str(psnr_values)]);

% Compare MSE between original-reconstructed images and original-quantized wavelet coefficients
disp('Mean Squared Error (MSE) Comparison:');
disp('MSE between original and reconstructed images after quantization:');
disp(mse_reconstructed);
disp('MSE between original and quantized wavelet coefficients:');
disp(mse_quantized_coeffs);

% Estimate bit-rate using the range of quantization step sizes
% Calculate bits per coefficient using logarithm with base 2
bits_per_coefficient = log2(256 ./ max(quantization_steps, 1)); % 256 represents the dynamic range of coefficients (8-bit image)

% Total coefficients = Total number of wavelet coefficients * bits per coefficient 
total_coefficients = numel(return_coefficients{1}) * length(quantization_steps); 

bit_rate = total_coefficients * bits_per_coefficient;

% Display or use bit_rate for further analysis, plotting, or visualization
disp('Estimated Bit-Rate for Each Quantization Step:');
disp(bit_rate);



function mse = calculateMSE(matrix1, matrix2)
    % Ensure matrices have the same size
    assert(isequal(size(matrix1), size(matrix2)), 'Matrices must have the same size.');

    % Calculate Mean Squared Error
    squaredError = (matrix1 - matrix2).^2;
    mse = mean(squaredError(:));
end

function psnr = calculate_psnr(quantization_step, original_boats_dct_coefficients, original_harbour_dct_coefficients, original_peppers_dct_coefficients)
    quantized_boats_coefficients = round(original_boats_dct_coefficients / quantization_step) * quantization_step;
    quantized_harbour_coefficients = round(original_harbour_dct_coefficients / quantization_step) * quantization_step;
    quantized_peppers_coefficients = round(original_peppers_dct_coefficients / quantization_step) * quantization_step;
    %total_mse = mean([calculateMSE(original_boats_image, restored_boats_image), calculateMSE(original_harbour_image, restored_harbour_image), calculateMSE(original_peppers_image, restored_peppers_image)]);
    total_mse = mean([calculateMSE(original_boats_dct_coefficients, quantized_boats_coefficients), calculateMSE(original_harbour_dct_coefficients, quantized_harbour_coefficients), calculateMSE(original_peppers_dct_coefficients, quantized_peppers_coefficients)]);

    psnr = 10*log10((255*255)/total_mse);
end

function total_entropy = calculate_entropy(quantization_step, original_boats_dct_coefficients, original_harbour_dct_coefficients, original_peppers_dct_coefficients)
    quantized_boats_coefficients = round(original_boats_dct_coefficients / quantization_step) * quantization_step;
    quantized_harbour_coefficients = round(original_harbour_dct_coefficients / quantization_step) * quantization_step;
    quantized_peppers_coefficients = round(original_peppers_dct_coefficients / quantization_step) * quantization_step;

    % Concatenate the matrices into a single column vector
    allValues = [quantized_boats_coefficients(:); quantized_harbour_coefficients(:); quantized_peppers_coefficients(:)];
    
    % Use histcounts to compute the frequency of each unique value
    uniqueValues = unique(allValues);
    counts = histcounts(allValues, [uniqueValues; max(uniqueValues)+1]);
    total_pixels = sum(counts);
    total_entropy = 0;
    probabilities = arrayfun(@(x) x/total_pixels ,counts);
    for i = 1:length(probabilities)
        current_element = probabilities(i);
        total_entropy = total_entropy - (current_element * log2(current_element));
    end
end

function processed_signal = analysis_filter_bank(signal, scaling_vector)
    % Time reversed signal
    signal = flip(signal);
    % Process the high band
    low_band = conv(signal, scaling_vector);
    low_band = low_band(1:length(signal));
    approximation = low_band(1:2:end);
    approximation = double(approximation);

    % Process the low band
    wavelet_vector = compute_wavelet_vector(scaling_vector);
    high_band = conv(signal, wavelet_vector);
    high_band = high_band(1:length(signal));
    detail = high_band(1:2:end);
    detail = double(detail);

    % Interleave the high band and low band together
    processed_signal = zeros(1,length(signal));
    % Add the approximation to the signal
    processed_signal(1:2:end) = detail;
    % Add the detail to the signal
    processed_signal(2:2:end) = approximation;
end

function reconstructed_signal = synthesis_filter_bank(wavelet_coeffs, scaling_vector)
    % Time reversed wavelet signal
    wavelet_coeffs = flip(wavelet_coeffs);

    % Periodic extend the signal
    signal_length = length(wavelet_coeffs);
    wavelet_coeffs = [wavelet_coeffs, wavelet_coeffs, wavelet_coeffs];

    % Upsample the high band
    high_band = wavelet_coeffs(2:2:end);
    upsampled_high_band = zeros(1, 2 * length(high_band));
    upsampled_high_band(1:2:end) = high_band;
    % Upsample the low band
    low_band = wavelet_coeffs(1:2:end);
    upsampled_low_band = zeros(1, 2*length(low_band));
    upsampled_low_band(1:2:end) = low_band;

    % Convolute the high band
    wavelet_vector = compute_wavelet_vector(scaling_vector);
    conv_high_band = conv(upsampled_high_band, wavelet_vector);
    %detail = conv_high_band(signal_length:1:(2*signal_length)-1);

    detail = circshift(conv_high_band, -signal_length);
    detail = detail(1, 1:signal_length);

    % Convolute the low band
    conv_low_band = conv(upsampled_low_band, scaling_vector);
    %approximation = conv_low_band(signal_length:1:(2*signal_length)-1);

    approximation = circshift(conv_low_band, -signal_length);
    approximation = approximation(1, 1:signal_length);

    % Return the signal
    reconstructed_signal = double(approximation+detail);
end

% Iterative analysis filter
function wavelet_coefficients = iterative_fwt(signal, scaling_vector, num_scales)
    % Initialize cell array to store wavelet coefficients
    wavelet_coefficients = cell(1, num_scales);

    % Initialize the signal for processing
    processed_signal = signal;

    % Iterate through the scales
    for scale = 1:num_scales
        % Apply analysis filter bank to obtain wavelet coefficients
         return_signal = analysis_filter_bank(processed_signal, scaling_vector);
         approximation = return_signal(1:2:end);
         detail = return_signal(2:2:end);
        % Store wavelet coefficients for the current scale
        % add the approximation to the first row and the detail to the
        % second row
        wavelet_coefficients{scale} = [approximation; detail];

        % Prepare the processed signal for the next scale by splitting the
        % approximation into lower bands
        processed_signal = approximation;
    end
end

function reconstructed_signal = iterative_inv_fwt(wavelet_coeffs, scaling_vector, num_scales)

    % Save the low band during the iteration, it is the only thing that is
    % being iteratively reconstructed
    low_band = [];
    % Iterate through wavelet coefficient matrix in reverse order
    for i=num_scales:-1:1
        % Check is this if the first step of the reconstruction, we have to
        % handle this differently
        if (i == num_scales)
            approximation = wavelet_coeffs{i}(1,1:end);
            detail = wavelet_coeffs{i}(2,1:end);
            % Combine the wavelet approximation and detail of the lower band
            % into a signle signal for reconstrucion of the lower band of the
            % higher scale
            wavelet_signal = zeros(1,length(approximation) + length(detail));
            wavelet_signal(1:2:end) = approximation;
            wavelet_signal(2:2:end) = detail;
    
            % Current reconstructed signal for this scale
            low_band = synthesis_filter_bank(wavelet_signal, scaling_vector);
        else
            detail = wavelet_coeffs{i}(2,1:end);
            wavelet_signal = zeros(1,length(detail)+length(low_band));
            wavelet_signal(1:2:end) = low_band;
            wavelet_signal(2:2:end) = detail;
            low_band = synthesis_filter_bank(wavelet_signal, scaling_vector);
        end
    end
    reconstructed_signal = low_band;
end

% Caluclating the wavelet vector from the scaling vector
function wavelet_vector = compute_wavelet_vector(scaling_vector)
    % Flip the scaling vector
    flipped_scaling_vector = fliplr(scaling_vector);
    
    % Create the wavelet vector by alternating elements
    wavelet_vector = flipped_scaling_vector;
    wavelet_vector(1:2:end) = -wavelet_vector(1:2:end);
end

%section 3.3
function quantized_coeffs = uniform_quantizer(coeffs, step_size)
    num_scales = numel(coeffs);
    quantized_coeffs = cell(1, num_scales);

    for i = 1:num_scales
        current_coeffs = coeffs{i};
        quantized_coeffs{i} = round(current_coeffs / step_size) * step_size;
    end
end

%section 3.4
% Function to calculate PSNR between original and reconstructed images
function psnr_value = calculate_psnr2(original, reconstructed)
    mse = calculate_mse(original, reconstructed);
    psnr_value = 10 * log10(255^2 / mse);
end

% Function to calculate mean squared error (MSE) between two images
function mse_value = calculate_mse(image1, image2)
    mse_value = mean((image1(:) - image2(:)).^2);
end