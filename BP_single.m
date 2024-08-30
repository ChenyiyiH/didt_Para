%% 标准BP
% 基于BP神经网络的预测算法
function [errorsum_bps1,R2_bps1,MSE_bps1,RMSE_bps1,net_bp]=BP_single(datatable,train_par)
% 清空环境变量（根据需要）
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

%% 数据提取及归一化
% 加载输入和输出数据
input = data.in;
output = data.out;

% 随机打乱数据索引
numSamples = size(input, 1);
randIdx = randperm(numSamples);
n_idx = size(input,1) * train_par;
% 划分训练集和测试集
trainIdx = randIdx(1:n_idx);
testIdx = randIdx(n_idx+1:end);

input_train = input(trainIdx, :)';
output_train = output(trainIdx, :)';
input_test = input(testIdx, :)';
output_test = output(testIdx, :)';

% 数据归一化
[inputn, inputps] = mapminmax(input_train,0.5,1);
[outputn, outputps] = mapminmax(output_train,0.5,1);

%% BP网络训练
% 创建BP神经网络
hiddenLayerSize = 25;  % 隐含层神经元数量
% net_bp = feedforwardnet(hiddenLayerSize,'trainrp');
% net_bp = fitnet(hiddenLayerSize,'trainbr');
net_bp = newff(inputn, outputn, [10, 5], {'logsig', 'purelin'}, 'trainbr', 'l2', 0.001);
% 配置训练参数
net_bp.trainParam.epochs = 10000;    % 训练轮次
net_bp.trainParam.lr = 0.1;         % 学习率
net_bp.trainParam.goal = 1e-8;      % 目标误差
% net_bp.trainParam.mu = 0.5;
net_bp.layers{1}.transferFCn = 'logsig';
net_bp.layers{2}.transferFCn = 'purelin';
net_bp.divideParam.trainRatio = 70/100;
net_bp.divideParam.valRatio = 20/100;
net_bp.divideParam.testRatio = 10/100;
% 训练网络
net_bp = train(net_bp, inputn, outputn,'useGPU','no');

%% BP网络预测
% 对测试数据进行归一化
inputn_test = mapminmax('apply', input_test, inputps);

% 网络预测
predictedOutput = sim(net_bp, inputn_test);

% 反归一化预测结果
BPoutput = mapminmax('reverse', predictedOutput, outputps);

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

% 预测误差
error_bps = BPoutput - output_test;

% 误差图
% figure;
% plot(error_bps, '-*');
% title('BP网络预测误差', 'FontSize', 12);
% xlabel('样本', 'FontSize', 12);
% ylabel('误差', 'FontSize', 12);

% 预测误差百分比图
% figure;
% plot((output_test - BPoutput) ./ BPoutput, '-*');
% title('神经网络预测误差百分比', 'FontSize', 12);

% 计算误差总和
errorsum_bps1 = sum(abs(error_bps));

%% 评估模型性能
% 决定系数 (R²)
R = corrcoef(output_test, BPoutput);
R2_bps1 = R(1, 2)^2;

% 均方误差 (MSE) 和均方根误差 (RMSE)
MSE_bps1 = immse(output_test, double(BPoutput));
RMSE_bps1 = sqrt(MSE_bps1);

% 打印结果
fprintf('R² = %.4f\n', R2_bps1);
fprintf('MSE = %.4f\n', MSE_bps1);
fprintf('RMSE = %.4f\n', RMSE_bps1);
end