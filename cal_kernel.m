function [k] = calcKernel_Orig2Blurred(img_orig,img_blurred,ksize,display)

kernelsz = ksize;    
Halfkernelsz = floor(kernelsz./2);
kernelsz = Halfkernelsz.*2+1; 
nPoints2use = 10000;
nSigmasAwayFromMean = 1; 
nBestKsToSelect = 1;
img = img_orig;

counter = 0;
for ii = -Halfkernelsz(1):Halfkernelsz(1)
    for jj = -Halfkernelsz(2):Halfkernelsz(2)
        counter = counter+1;
        partialImg =  img(Halfkernelsz(1)+ii+1:size(img,1)-Halfkernelsz(1)+ii,...
                        Halfkernelsz(2)+jj+1:size(img,2)-Halfkernelsz(2)+jj);
        B(1:numel(partialImg),counter) = partialImg(:);
    end
end
A = img_blurred(Halfkernelsz(1)+1:size(img,1)-Halfkernelsz(1),...
                        Halfkernelsz(2)+1:size(img,2)-Halfkernelsz(2));
A_vec = A(:);  
rand = randperm(size(A_vec,1));
C = B(rand(1:nPoints2use),:);
d = A_vec(rand(1:nPoints2use));
x = lsqnonneg(C,d);
x = x/sum(x);
k = reshape(x,[kernelsz(1),kernelsz(2),nBestKsToSelect]);


if nargout > 1 || display
%% reproduce image
img_deblurreds = zeros(size(img_blurred));
if nBestKsToSelect > 1
    for ii = 1:nBestKsToSelect
          img_deblurreds(:,:,ii) = deconv_fast_mask(edgetaper(im2double(img_blurred),Ks(:,:,ii)),Ks(:,:,ii));
    end
    img_deblurred = mode(img_deblurreds,3);
else
    img_deblurred = deconv_fast_mask(edgetaper(im2double(img_blurred),k),k);
end
if (display)
    varargout{1} = img_deblurred;
    figure; imshow([img_orig,img_blurred,img_deblurred],[]) % if you want preview the output
    title ( 'original image, blurred image, deblurred image');
end

end


end

% functions Aux 
function [im_out] = deconv_fast_mask(im_blurred,kernel)
% addpath(genpath('.\Fortunato_Oliveira_FD_Script_our_method_only\our_method\'));

%% prep variables
small_kernel = kernel;
KR = floor((size(small_kernel, 1) - 1)/2); 
KC = floor((size(small_kernel, 2) - 1)/2); 
pad_size = 2 * max(KR, KC);
wev   = [0.001, 20, 0.033, 0.05]; 

%% prep img and kernel
[im_blurred_padded, mask_pad] = imPad(im_blurred, pad_size);    
[R, C, CH] = size(im_blurred_padded);   
big_kernel = getBigKernel(R, C, small_kernel);

%% get deconvolved img
im_out_padded = our_method_bifilter(im_blurred_padded, big_kernel, wev);       
im_out        = imUnpad(im_out_padded, mask_pad, pad_size);

end
function kernel = getBigKernel(R, C, small_kernel)
% Resize kernel to match image size

kernel = zeros(R,C); 
RC     = floor(R/2); 
CC     = floor(C/2); 

[RF,CF] = size(small_kernel); 
RCF = floor(RF/2); CCF = floor(CF/2); 

kernel(RC-RCF+1:RC-RCF+RF,CC-CCF+1:CC-CCF+CF) = small_kernel;
kernel = ifftshift(kernel);
kernel = kernel ./ sum(kernel(:));
end
function [im_out, mask] = imPad(im_in, pad)
% Pad image to remove border ringing atifacts (see section 4.1 of our paper)

im_pad   = padarray(im_in, [pad, pad],'replicate','both');
[R,C,CH] = size(im_pad);

[X Y] = meshgrid (1:C, 1:R);

X0 = 1 + floor ( C / 2); Y0 = 1 + floor ( R / 2);
DX = abs( X - X0 )     ; DY = abs( Y - Y0 );
C0 = X0 - pad          ; R0 = Y0 - pad;

alpha = 0.01;
% force mask value at the borders aprox equal to alpha
% this makes the transition smoother for large kernels
nx = ceil(0.5 * log((1-alpha)/alpha) / log(X0 / C0));
ny = ceil(0.5 * log((1-alpha)/alpha) / log(Y0 / R0));

mX = 1 ./ ( 1 + ( DX ./ C0 ).^ (2 * nx));
mY = 1 ./ ( 1 + ( DY ./ R0 ).^ (2 * ny));
mask_0 = mX .* mY;

mask   = zeros(R,C,CH);
for ch = 1:CH
    mask(:,:,ch) = mask_0;
end;

im_out = zeros(R,C,CH);
im_out = im_pad .* mask;
end
function im_out = imUnpad(im_in, mask_pad, pad)
% Remove padding (see section 4.1 of our paper)

im_out1 = im_in ./ mask_pad;
im_out = im_out1(pad+1:end-pad, pad+1:end-pad, :);
end
