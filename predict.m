function [y1] = predict(x1)
%PREDICT neural network simulation function.
%
% Auto-generated by MATLAB, 30-Aug-2024 15:17:32.
% 
% [y1] = predict(x1) takes these arguments:
%   x = 2xQ matrix, input #1
% and returns:
%   y = 2xQ matrix, output #1
% where Q is the number of samples.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [619.962003338752;4.43990629286854e-06];
x1_step1.gain = [0.00196028267295341;122727.657587669];
x1_step1.ymin = -1;

% Layer 1
b1 = [-2.8729555318025119526;-1.5201452825676524672;0.015049231459093287411;-1.0651166940904530644;-2.9883018093273854277];
IW1_1 = [-1.0648695775476610947 -2.5714530488150466603;3.3656140045511011571 3.480480442059681323;-1.6960155560266501062 -1.6830234061560911396;-2.7179084520989094109 0.70786117560251737846;-1.5228036352501921336 2.5464712946626653078];

% Layer 2
b2 = [-0.47002148368542728818;-0.14331110502615770907];
LW2_1 = [-0.6540776496607630941 0.42018398997659800465 -0.77214835078705768012 -0.26341475671226666222 -0.28195775438888565079;-0.53727168203020370107 0.062365722221025744754 -0.29715400242518630325 0.70326550225044937026 0.27360876040917225804];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [4.08163265306123;51282.0512820513];
y1_step1.xoffset = [0.2;1e-05];

% ===== SIMULATION ========

% Dimensions
Q = size(x1,2); % samples

% Input 1
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = repmat(b2,1,Q) + LW2_1*a1;

% Output 1
y1 = mapminmax_reverse(a2,y1_step1);
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
  y = bsxfun(@minus,x,settings.xoffset);
  y = bsxfun(@times,y,settings.gain);
  y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
  x = bsxfun(@minus,y,settings.ymin);
  x = bsxfun(@rdivide,x,settings.gain);
  x = bsxfun(@plus,x,settings.xoffset);
end