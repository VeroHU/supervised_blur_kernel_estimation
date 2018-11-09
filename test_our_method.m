% test_our_method
% Parameters: none%
%
% deconvolution method usage examples for: 
% Fast High-Quality non-Blind Deconvolution Using Sparse Adaptive Priors algorithm
% 
%  This is part of the reference implementation of the deconvolution method
%  described in the paper:
% 
%  Fast High-Quality non-Blind Deconvolution Using Sparse Adaptive Priors
%  Horacio E. Fortunato and Manuel M. Oliveira 
%  The Visual Computer,
%  Springer, Volume 30, Numbers 6-8, June 2014. pp. 661-671. ISSN 0178-2789 (Print) 1432-2315 (Online) 
%  DOI: 10.1007/s00371-014-0966-x
%  (2nd Best Paper Award at Computer Graphics International 2014)
%
%  Please refer to the publication above if you use this software. 
%
%  THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY EXPRESSED OR IMPLIED WARRANTIES
%  OF ANY KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
%  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
%  OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
%  THIS SOFTWARE.
%
%  Version 1.0 - Oct. 2014.
%
% ===================================================
% 					Acknowledgements:
% ===================================================
%
%  Edge preserving filter implementation from:
% 
%    Domain Transform for Edge-Aware Image and Video Processing
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 30 (2011), Number 4.
%    Proceedings of SIGGRAPH 2011, Article 69.
%
% ===================================================
% Synthetic image example
%
%   Photograph from: 
%     'Kodak Lossless True Color Image Suite'.
%
%   Kernel from: 
%     Krishnan, D., and Fergus, R. 2009. Fast image deconvolution 
%     using hyper-laplacian priors. In Advances in Neural Information 
%     Processing Systems 22, 1033-1041.
%
% ===================================================
% Natural images examples
%
%   Blurry photographs with corresponding camera-motion kernels from:
%     Shan, Q., Jia, J., and Agarwala, A. 2008. High-quality motion deblurring 
%     from a single image. ACM TOG 27
%
% ===================================================

function test_our_method()  

path ('our_method', path);
clc;
clear all;
close all;

% ==========================================================================
%                        Example 01: Synthetic 
% ==========================================================================
% 
im_in_orig = im2double(imread('./images/kodim03.png'));
% load kernel 
load  './images/Krishnan_kernel1.mat';
filt = kernel1;
small_kernel = im2double(filt);   

% add blurr and noise
sigma   = 0.01; % noise standard deviation
[im_blurred, im_in_valid, noise] = blurrAndNoiseLinear(im_in_orig, small_kernel, sigma);
    
% ==========================================================================
KR = floor((size(small_kernel, 1) - 1)/2); 
KC = floor((size(small_kernel, 2) - 1)/2); 
SNR_pad_size = max(KR,KC); 
pad_size = 2 * max(KR, KC);

snr_blur  = our_snr(im_blurred, SNR_pad_size, im_in_valid);
  
wev   = [0.001, 20, 0.033, 0.05]; 

[im_blurred_padded, mask_pad] = imPad(im_blurred, pad_size);    
[R, C, CH] = size(im_blurred_padded);   
big_kernel = getBigKernel(R, C, small_kernel);

%--------------------------------------------------------------------------
im_out_padded = our_method_bifilter(im_blurred_padded, big_kernel, wev);       
im_out        = imUnpad(im_out_padded, mask_pad, pad_size);
%--------------------------------------------------------------------------
snr_out  = our_snr(im_out, SNR_pad_size, im_in_valid);
   
disp('------------------------------------');
disp(sprintf('\nPSNR:\n',snr_blur, snr_out));
disp(sprintf('\nblurred     :%6.3f\n',snr_blur));
disp(sprintf('\ndeconvolved :%6.3f\n',snr_out));

figure; 
subplot(3,3,1);imshow(im_blurred );title('blurred'); 
subplot(3,3,2);imshow(im_out);title('out'); 
subplot(3,3,3);imshow(mat2gray(small_kernel)); title('small kernel');drawnow;

% ==========================================================================
%           Example 02: Motion blurr, natural image and kernel
% ==========================================================================
 
im_blurred  = im2double(imread('./images/test4.bmp'));
im_kernel   = im2double(imread('./images/kernel4.bmp'));
small_kernel = im_kernel(:,:,1);
wev   = [0.001, 20, 0.033, 0.05]; 
% ==========================================================================
KR = floor((size(small_kernel, 1) - 1)/2); 
KC = floor((size(small_kernel, 2) - 1)/2); 
pad_size = 2 * max(KR, KC);

[im_blurred_padded, mask_pad] = imPad(im_blurred, pad_size);    
[R, C, CH] = size(im_blurred_padded);   
big_kernel = getBigKernel(R, C, small_kernel);

%--------------------------------------------------------------------------
im_out_padded = our_method_bifilter(im_blurred_padded, big_kernel, wev);       
im_out        = imUnpad(im_out_padded, mask_pad, pad_size);
%--------------------------------------------------------------------------
   
subplot(3,3,4);imshow(im_blurred );title('blurred'); 
subplot(3,3,5);imshow(im_out);title('out'); 
subplot(3,3,6);imshow(small_kernel); title('small kernel');drawnow;

% ==========================================================================
%           Example 03: Motion blurr, natural image and kernel
% ==========================================================================
im_blurred   = im2double(imread('./images/redTreeBlurImage.png'));
im_kernel    = im2double(imread('./images/outKernelRT.png'));
small_kernel = im_kernel(:,:,1);
wev   = [0.0001, 20, 0.01, 0.01];
% ==========================================================================
KR = floor((size(small_kernel, 1) - 1)/2); 
KC = floor((size(small_kernel, 2) - 1)/2); 
pad_size = 2 * max(KR,KC);

[im_blurred_padded, mask_pad] = imPad(im_blurred, pad_size);    
[R, C, CH] = size(im_blurred_padded);   
big_kernel = getBigKernel(R, C, small_kernel);

%--------------------------------------------------------------------------
im_out_padded = our_method_bifilter(im_blurred_padded, big_kernel, wev);       
im_out        = imUnpad(im_out_padded, mask_pad, pad_size);
%--------------------------------------------------------------------------
   
subplot(3,3,7);imshow(im_blurred );title('blurred'); 
subplot(3,3,8);imshow(im_out);title('out'); 
subplot(3,3,9);imshow(small_kernel); title('small kernel');drawnow;

% ==========================================================================
%                           Aux. functions
% ==========================================================================
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
% ==========================================================================

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

% ==========================================================================
function im_out = imUnpad(im_in, mask_pad, pad)
% Remove padding (see section 4.1 of our paper)

im_out1 = im_in ./ mask_pad;
im_out = im_out1(pad+1:end-pad, pad+1:end-pad, :);

% ==========================================================================
% Blurr image and add noise for synthetic example (linear convolution)
function [im_blurred, im_in_valid, noise] = blurrAndNoiseLinear(im_in, small_kernel, sigma)
[R, C, CH] = size(im_in);

    for ch = 1:CH
        im_blurred(:,:,ch) = conv2(im_in(:,:,ch), small_kernel, 'valid');
    end;
    
    [RB, CB, CH] = size(im_blurred);
    
    noise = randn(RB, CB, CH) * sigma;
    im_blurred = im_blurred + noise; 
    im_blurred = double(uint8(im_blurred .* 255))./255;
    
    RB2 = floor((R-RB)/2); CB2 = floor((C-CB)/2);
    im_in_valid = im_in(RB2+1:RB2+RB, CB2+1:CB2+CB, :);
 
% ==========================================================================
function PSNR = our_snr(sig, hk, ref)

sig = double(uint8(sig .* 255))./255;
ref = double(uint8(ref .* 255))./255;

ref2  = ref  (hk+1:end-hk, hk+1:end-hk, :);
sig2  = sig  (hk+1:end-hk, hk+1:end-hk, :);

mse  = mean ((ref2(:) - sig2(:)).^2);
PSNR = 10 * log10( 1 / mse);
