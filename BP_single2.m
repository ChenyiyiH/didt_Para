% 清空环境变量（根据需要）
function [errorsum_bps2,R2_bps2,MSE_bps2,RMSE_bps2,net_bp]=BP_single2(datatable,train_par)
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);
% 训练数据和预测数据提取及归一化
input = data.in;
output = data.out;
n = randperm(size(input, 1));  % 使用 size(input, 1) 代替硬编码的 16000
n_idx = size(input,1) * train_par;
% 划分训练集和测试集
train_idx = 1:n_idx;
test_idx = n_idx+1:size(input, 1);

input_train = input(n(train_idx), :)';
output_train = output(n(train_idx), :)';
input_test = input(n(test_idx), :)';
output_test = output(n(test_idx), :)';

% 数据归一化
[inputn, inputps] = mapminmax(input_train);
[outputn, outputps] = mapminmax(output_train);

%% BP网络训练
% 创建BP网络，隐藏层有15个神经元
net_bp = feedforwardnet(15,'trainscg'); 
net_bp.trainParam.epochs = 1000; % 训练次数
net_bp.trainParam.lr = 0.01;      % 学习率
net_bp.trainParam.goal = 1e-4;   % 训练目标误差

% 训练网络
net_bp = train(net_bp, inputn, outputn);

%% BP网络预测
% 对测试数据进行归一化
inputn_test = mapminmax('apply', input_test, inputps);
% 网络仿真
an = sim(net_bp, inputn_test);
% 反归一化预测结果
BPoutput = mapminmax('reverse', an, outputps);

%% 结果分析
% 预测输出图
% figure;
% plot(BPoutput, ':og', 'DisplayName', '预测输出');
% hold on;
% plot(output_test, '-*', 'DisplayName', '期望输出');
% legend('show');
% title('BP网络预测输出', 'FontSize', 12);
% xlabel('样本', 'FontSize', 12);
% ylabel('函数输出', 'FontSize', 12);

% 计算预测误差
error_bps2 = BPoutput - output_test;

% 误差图
% figure;
% plot(error_bps2, '-*');
% title('BP网络预测误差', 'FontSize', 12);
% xlabel('样本', 'FontSize', 12);
% ylabel('误差', 'FontSize', 12);
% 
% % 预测误差百分比图
% figure;
% plot((output_test - BPoutput) ./ BPoutput, '-*');
% title('神经网络预测误差百分比', 'FontSize', 12);

% 计算误差总和
errorsum_bps2 = sum(abs(error_bps2));

%% 评估模型性能
% 决定系数 (R²)
R_bps2 = corrcoef(output_test, BPoutput);
R2_bps2 = R_bps2(1, 2)^2;

% 均方误差和均方根误差
MSE_bps2 = immse(output_test, double(BPoutput));
RMSE_bps2 = sqrt(MSE_bps2);

% 打印结果
fprintf('R² = %.4f\n', R2_bps2);
fprintf('MSE = %.4f\n', MSE_bps2);
fprintf('RMSE = %.4f\n', RMSE_bps2);
end