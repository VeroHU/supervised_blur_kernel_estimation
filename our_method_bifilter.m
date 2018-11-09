% our_method_bifilter
% Fast High-Quality non-Blind Deconvolution Using Sparse Adaptive Priors
%========================================================================
% 
%  Parameters:
%    im_blurred:  Input blurred and padded image (B/W or color)
%    kernel:      Kernel (same size as image, B/W)
%    we:          array with 4 parameters:
%      we(1) : Regularization parameter for gaussian step 
%      we(2) and we(3) :  Bilateral filter parameters, sigma_s = we(2), sigma_r = we(3) ,
%                        (spatial and range standard deviation)
%      we(4) : Regularization parameter for actual deconvolution (final step)
%
%  This is the reference implementation of the deconvolution method
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

function im_out = our_method_bifilter(im_blurred, kernel, we)

[R, C, CH] = size(im_blurred);

im_out      = zeros(R, C, CH);
im_out_temp = zeros(R, C, CH);
im_out_bi   = zeros(R, C, CH);
im_out_bi2  = zeros(R, C, CH);
B0          = zeros(R, C, CH);

fft2_kernel = fft2(kernel);
conj_fft2_kernel = conj(fft2_kernel);
A0 = real(conj_fft2_kernel .* fft2_kernel);

%clear fft2_kernel;
A1 = getA1(R,C);
A10  = A0 + we(1) .* A1;

% Gaussian step
for ch = 1:CH % RGB channels
    B0(:,:,ch)  = conj_fft2_kernel .* fft2(im_blurred(:,:,ch));
    im_out_temp(:,:,ch)   = real(ifft2(B0(:,:,ch) ./ A10)); 
end;

% Bilateral filter:
sigma_s = we(2);  sigma_r = we(3);

im_out_bi = our_RF(im_out_temp, sigma_s, sigma_r);

% Actual deconvolution

A12  = A0 + we(4) .* A1;
for ch = 1:CH % RGB channels
    b1 = get_b1(im_out_bi(:,:,ch));
    B = B0(:,:,ch) + we(4) .* fft2(b1);    
    im_out(:,:,ch)  = real(ifft2(B ./ A12)); 
end;

% ===================================================

% ===================================================
function A1 = getA1(R,C)
% A1:
a1 = zeros(R, C);

% use_dx
    a1(1  , 1) = 1;
    a1(1  , 3) = -0.25; a1(1  ,C-1) = -0.25;
    a1(R-1, 1) = -0.25; a1(3  ,1  ) = -0.25;

% use_dxx
    a1(1,  1) = a1(1,  1) + 6;
    a1(1,  2) = a1(1,  2) - 2;
    a1(1,  C) = a1(1,  C) - 2;
    a1(R,  1) = a1(R,  1) - 2;
    a1(2,  1) = a1(2,  1) - 2;
    a1(1,  3) = a1(1,  3) + 0.5;
    a1(1,C-1) = a1(1,C-1) + 0.5;
    a1(R-1,1) = a1(R-1,1) + 0.5;
    a1(3,  1) = a1(3,  1) + 0.5;

% use_dxy
    a1(1, 1) = a1(1, 1) + 8;
    a1(1, 2) = a1(1, 2) - 4;
    a1(1, C) = a1(1, C) - 4;
    a1(R, 1) = a1(R, 1) - 4;
    a1(2, 1) = a1(2, 1) - 4;
    a1(2, 2) = a1(2, 2) + 2;
    a1(R, C) = a1(R, C) + 2;
    a1(R, 2) = a1(R, 2) + 2;
    a1(2, C) = a1(2, C) + 2;

 A1 = real(fft2(a1));

% ===================================================

function b1 = get_b1(im_in)

dx   = [-0.5, 0, 0.5];
dy   = [-0.5; 0; 0.5];
dxx  = [-1 / 1.4142, 2 / 1.4142, -1 / 1.4142];
dyy  = [-1 / 1.4142; 2 / 1.4142; -1 / 1.4142]; 
dxy  = [-1.4142, 1.4142, 0; 1.4142, -1.4142, 0 ; 0, 0, 0];

dx  = rot90(dx,2 ); %rot 180 deg, same as flipud(fliplr(dx));    
dy  = rot90(dy,2 );
dxx = rot90(dxx,2);
dyy = rot90(dyy,2);
dxy = rot90(dxy,2); 

conj_dx  = rot90(dx,2 ); %rot 180 deg, same as flipud(fliplr(dx));    
conj_dy  = rot90(dy,2 );
conj_dxx = rot90(dxx,2);
conj_dyy = rot90(dyy,2);
conj_dxy = rot90(dxy,2);    

% use_dx
lambda = 0.065;

w  = conv2(im_in, dx, 'same');       
w  = sparse(w, lambda);
b1 = conv2(w, conj_dx, 'same');

w  = conv2(im_in, dy, 'same');
w  = sparse(w, lambda);
b1 = b1 + conv2(w, conj_dy, 'same');

% use_dxx
lambda = 0.5 * lambda;

w  = conv2(im_in, dxx, 'same');
w  = sparse(w, lambda);
b1 = b1 + conv2(w, conj_dxx, 'same');

w  = conv2(im_in, dyy, 'same');
w  = sparse(w, lambda);
b1 = b1 + conv2(w, conj_dyy, 'same');

% use_dxy
w  = conv2(im_in, dxy, 'same');
w  = sparse(w, lambda);
b1 = b1 + conv2(w, conj_dxy, 'same');  % b1xy
        
% ===================================================

function dw = sparse(x, lambda)
%dw =  x .* ( 1 - (1 ./ (1 + ((x / lambda) .^ 4)) ) );
    
    xl = lambda ./ x;
    xl = xl .* xl;
    xl = xl .* xl; % = (lambda ./ x) .^ 4
    xl = 1 + xl;
    
    dw =  x ./ xl;

    
