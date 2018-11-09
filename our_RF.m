%========================================================================
%  THIS IS NOT THE ORIGINAL IMPLEMENTATION 
%========================================================================
%
%  Modified by: Horacio E. Fortunato - 28/11/2013
% - Use Sobel filters to compute the l1-norm distance of neighbor pixels.
% - Only one iteration 
% - Do not use joint_image

%========================================================================
%========================================================================
% Original text:
%========================================================================
%  RF  Domain transform recursive edge-preserving filter.
% 
%  F = RF(img, sigma_s, sigma_r, num_iterations, joint_image)
%
%  Parameters:
%    img             Input image to be filtered.
%    sigma_s         Filter spatial standard deviation.
%    sigma_r         Filter range standard deviation.
%    num_iterations  Number of iterations to perform (default: 3).
%    joint_image     Optional image for joint filtering.
%
%
%
%  This is the reference implementation of the domain transform RF filter
%  described in the paper:
% 
%    Domain Transform for Edge-Aware Image and Video Processing
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 30 (2011), Number 4.
%    Proceedings of SIGGRAPH 2011, Article 69.
%
%  Please refer to the publication above if you use this software. For an
%  up-to-date version go to:
%  _old
%             http://inf.ufrgs.br/~eslgastal/DomainTransform/
%
%
%  THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY EXPRESSED OR IMPLIED WARRANTIES
%  OF ANY KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
%  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
%  OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
%  THIS SOFTWARE.
%
%  Version 1.0 - August 2011.

function im_out = our_RF(im_in, sigma_s, sigma_r)

    [R C CH] = size(im_in);
     
    % Compute the domain transform (Equation 11 of our paper).
    % Estimate horizontal and vertical partial derivatives using finite
    % differences.
            
    dIdx = zeros(R,C);
    dIdy = zeros(R,C);
    
    % Compute the l1-norm distance of neighbor pixels.
    % use sobel filters matrices
    dx = [-1, 0, 1;
          -2, 0, 2;
          -1, 0, 1] / 8.0;   

    dy = [-1, -2, -1;
           0,  0,  0;
           1,  2,  1] / 8.0;
   
    for ch = 1:CH
% original work derivatives        
%       dIdx(:,2:end) = dIdx(:,2:end) + abs( dIcdx(:,:,ch) );
%       dIdy(2:end,:) = dIdy(2:end,:) + abs( dIcdy(:,:,ch) );

        dIcdx(:,:,ch) = conv2(im_in(:,:,ch) , dx, 'same');  
        dIcdy(:,:,ch) = conv2(im_in(:,:,ch) , dy, 'same');  
        dIdx = dIdx + abs( dIcdx(:,:,ch) );
        dIdy = dIdy + abs( dIcdy(:,:,ch) );
    end
    
    dIdx = dIdx / CH;
    dIdy = dIdy / CH;
        
    % Compute the derivatives of the horizontal and vertical domain transforms.
    dHdx = (1 + sigma_s/sigma_r * dIdx);
    dVdy = (1 + sigma_s/sigma_r * dIdy);
       
    % The vertical pass is performed using a transposed image.
    dVdy = dVdy';
    % Perform the filtering.
    im_out = image_transpose(im_in);
    im_out = TransformedDomainRecursiveFilter_Horizontal(im_out, dVdy, sigma_s);

    im_out = image_transpose(im_out);
    im_out = TransformedDomainRecursiveFilter_Horizontal(im_out, dHdx, sigma_s);
 
end

% Recursive filter.
function F = TransformedDomainRecursiveFilter_Horizontal(im_in, D, sigma_s)

    % Feedback coefficient (Appendix of our paper).
    a = exp(-sqrt(2) / sigma_s);
    
    F = im_in;
    V = a .^ D;
    
   [R C CH] = size(im_in);  
    
    % Left -> Right filter.
    for i = 2:C
        for ch = 1:CH
            F(:,i,ch) = F(:,i,ch) + V(:,i) .* ( F(:,i - 1,ch) - F(:,i,ch) );
        end
    end
    
    % Right -> Left filter.
    for i = C-1:-1:1
        for ch = 1:CH
            F(:,i,ch) = F(:,i,ch) + V(:,i+1) .* ( F(:,i + 1,ch) - F(:,i,ch) );
        end
    end

end

% Recursive filter.
%
function im_out = image_transpose(im_in)

    [R C CH] = size(im_in);   
    im_out = zeros([C, R, CH], class(im_in));
    
    for ch = 1:CH
        im_out(:,:,ch) = im_in(:,:,ch)';
    end
    
end
