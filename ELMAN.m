%% 基于Elman神经网络的预测模型研究
function [errorsum_elman,R2_elman,MSE_elman,RMSE_elman,net_elman]=ELMAN(datatable,train_par)
%% 清空环境变量
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

%% 数据载入
% 载入数据并将数据分成训练和测试集
input = data.in;   % 16000x3
output = data.out; % 16000x2

% 打乱数据
n = randperm(size(input, 1));
n_idx = size(input,1) * train_par;
% 划分训练集和测试集
input_train = input(n(1:n_idx), :); % 13000x3
output_train = output(n(1:n_idx), :); % 13000x2
input_test = input(n(n_idx+1:end), :); % 3000x3
output_test = output(n(n_idx+1:end), :); % 3000x2

% 数据归一化
[inputn, inputps] = mapminmax(input_train'); % 归一化到[0, 1]
[outputn, outputps] = mapminmax(output_train'); % 归一化到[0, 1]
inputn_test = mapminmax('apply', input_test', inputps); % 归一化测试数据
output_testn = mapminmax('apply', output_test', outputps); % 归一化测试数据输出

% 确保数据为双精度
inputn = double(inputn);
outputn = double(outputn);
inputn_test = double(inputn_test);
output_testn = double(output_testn);

%% 选取训练数据和测试数据
% 训练数据输入
p_train = inputn;
% 训练数据输出
t_train = outputn;
% 测试数据输入
p_test = inputn_test;
% 测试数据输出
t_test = output_testn;

% 确保数据的维度一致
assert(size(p_train, 2) == size(t_train, 2), '训练数据输入和输出的样本数不一致');
assert(size(p_test, 2) == size(t_test, 2), '测试数据输入和输出的样本数不一致');

%% 网络的建立和训练
% 不同隐藏层神经元个数的设置
% nn = [7, 11, 14, 18];%
nn= 15;

% 误差记录初始化
error = zeros(length(nn), size(t_test, 2));

for i = 1:length(nn)
    % 建立Elman神经网络 隐藏层为nn(i)个神经元
    % 这里创建的网络结构：输入层 - 隐藏层 - 输出层
    net_elman = newelm(minmax(p_train), [nn(i) size(t_train, 1)], {'tansig', 'purelin'});

    % 设置网络训练参数
    net_elman.trainParam.epochs = 2000;
    net_elman.trainParam.show = 10;
    net_elman.trainParam.goal = 1e-5;

    % 初始化网络
    net_elman = init(net_elman);

    % Elman网络训练
    net_elman = train(net_elman, p_train, t_train);

    % 预测数据
    y_elman = sim(net_elman, p_test);

    % 反归一化
    y_sim_elman = mapminmax('reverse', y_elman, outputps); % 反归一化

    % 计算均方根误差
    error(i, :) = sqrt(mean((y_sim_elman - output_test').^2, 1)); % 计算均方根误差
end

%% 通过作图观察不同隐藏层神经元个数时，网络的预测效果
% figure;
% plot(1:size(error, 2), error(1, :), '-ro', 'LineWidth', 2);
% hold on;
% plot(1:size(error, 2), error(2, :), 'b:x', 'LineWidth', 2);
% hold on;
% plot(1:size(error, 2), error(3, :), 'k-.s', 'LineWidth', 2);
% hold on;
% plot(1:size(error, 2), error(4, :), 'c--d', 'LineWidth', 2);
% title('Elman预测误差图');
% legend('7', '11', '14', '18', 'Location', 'best');
% xlabel('时间点');
% ylabel('均方根误差');
% grid on;
% hold off;
%%
% 评估模型性能
error_elman = y_sim_elman' - output_test;

% 计算总误差
errorsum_elman = sum(abs(error_elman));

R_elman = corrcoef(output_test, y_sim_elman');
R2_elman = R_elman(1, 2)^2;  % 决定系数 (R²)

MSE_elman = immse(output_test', double(y_sim_elman));  % 均方误差
RMSE_elman = sqrt(MSE_elman);  % 均方根误差

% 打印结果
disp(['R² = ', num2str(R2_elman)]);
disp(['MSE = ', num2str(MSE_elman)]);
disp(['RMSE = ', num2str(RMSE_elman)]);