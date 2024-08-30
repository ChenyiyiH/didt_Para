%% 基于双隐含层BP神经网络的预测代码
function [errorsum,R2_bpd1,MSE_bpd1,RMSE_bpd1,net_bpdou]=BP_double(datatable,train_par)
% 清空环境变量（可根据需要选择性使用）
% clc;
% clear;
% close all;

nntwarn off;
load(datatable);

%% 数据处理
% 下载输入输出数据
input = data.in;
output = data.out;
n_idx = size(input,1) * train_par;
% 随机排序数据索引
numSamples = size(input, 1);
randIdx = randperm(numSamples);

% 划分训练集和测试集
trainIdx = randIdx(1:n_idx);
testIdx = randIdx(n_idx+1:end);

input_train = input(trainIdx, :)';
output_train = output(trainIdx);
input_test = input(testIdx, :)';
output_test = output(testIdx);

% 数据归一化
[inputn, inputps] = mapminmax(input_train);
[outputn, outputps] = mapminmax(output_train);

%% 创建和训练双隐含层BP神经网络
% 初始化网络结构
hiddenLayerSizes = [15 10];
% hiddenLayerSizes = 10;
net_bpdou = feedforwardnet(hiddenLayerSizes);

% 配置网络参数
net_bpdou.trainParam.epochs = 1000;
net_bpdou.trainParam.lr = 0.01;
net_bpdou.trainParam.goal = 1e-4; % 使用科学记数法简化

% 网络训练
net_bpdou = train(net_bpdou, inputn, outputn);

%% 网络预测
% 预测数据归一化
inputn_test = mapminmax('apply', input_test, inputps);

% 网络预测输出
predicted = sim(net_bpdou, inputn_test);

% 反归一化
BPoutput = mapminmax('reverse', predicted, outputps);

%% 结果分析
% 预测结果和实际结果的可视化
figure;
plot(BPoutput, ':og');
hold on;
plot(output_test, '-*');
legend('预测输出', '期望输出');
title('BP网络预测输出', 'FontSize', 12);
ylabel('函数输出', 'FontSize', 12);
xlabel('样本', 'FontSize', 12);

% 预测误差
error_bpd1 = BPoutput - output_test;

figure;
plot(error_bpd1, '-*');
title('BP网络预测误差', 'FontSize', 12);
ylabel('误差', 'FontSize', 12);
xlabel('样本', 'FontSize', 12);

% 预测误差百分比
figure;
plot((output_test - BPoutput) ./ BPoutput, '-*');
title('神经网络预测误差百分比');

% 计算总误差
errorsum = sum(abs(error_bpd1));

%% 评估模型性能
% 决定系数 (R²)
R_bpd1 = corrcoef(output_test, BPoutput);
R2_bpd1 = R_bpd1(1, 2)^2;

% 均方误差和均方根误差
MSE_bpd1 = immse(output_test, double(BPoutput));
RMSE_bpd1 = sqrt(MSE_bpd1);

% 打印结果
fprintf('R² = %.4f\n', R2_bpd1);
fprintf('MSE = %.4f\n', MSE_bpd1);
fprintf('RMSE = %.4f\n', RMSE_bpd1);
end