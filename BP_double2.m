% 清空环境变量（根据需要）
function [errorsum,R2_bpd2,MSE_bpd2,RMSE_bpd2,net_bpdou2]=BP_double2(datatable,train_par)
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);

% 训练数据和预测数据提取及归一化
input = data.in;
output = data.out;
n = randperm(size(input, 1));  % 使用 size(input, 1) 替代硬编码的 size
n_idx = size(input,1) * train_par;
% 划分训练集和测试集
input_train = input(n(1:n_idx), :)';
output_train = output(n(1:n_idx), :)';
input_test = input(n(n_idx+1:end), :)';
output_test = output(n(n_idx+1:end), :)';

% 数据归一化
[inputn, inputps] = mapminmax(input_train);
[outputn, outputps] = mapminmax(output_train);

% BP网络训练
net_bpdou2 = feedforwardnet([15 10],'trainrp'); % 双隐含层，分别有15和10个神经元
net_bpdou2.trainParam.epochs = 1000; % 训练次数
net_bpdou2.trainParam.lr = 0.1;
net_bpdou2.trainParam.goal = 1e-4; % 使用科学计数法更具语义
net_bpdou2 = train(net_bpdou2, inputn, outputn);

% BP网络预测
inputn_test = mapminmax('apply', input_test, inputps);
an = sim(net_bpdou2, inputn_test);
BPoutput = mapminmax('reverse', an, outputps);

% 结果分析
% figure;
% plot(BPoutput, ':og', 'DisplayName', '预测输出');
% hold on;
% plot(output_test, '-*', 'DisplayName', '期望输出');
% legend('show');
% title('BP网络预测输出', 'FontSize', 12);
% ylabel('函数输出', 'FontSize', 12);
% xlabel('样本', 'FontSize', 12);
% 
% % 预测误差
error_bpd2 = BPoutput - output_test;
% 
figure;
plot(error_bpd2, '-*');
title('BP网络预测误差', 'FontSize', 12);
ylabel('误差', 'FontSize', 12);
xlabel('样本', 'FontSize', 12);

figure;
plot((output_test - BPoutput) ./ BPoutput, '-*');
title('神经网络预测误差百分比', 'FontSize', 12);

% 计算误差总和
errorsum = sum(abs(error_bpd2));

% 评估模型性能
R_bpd2 = corrcoef(output_test, BPoutput);
R2_bpd2 = R_bpd2(1, 2)^2;  % 决定系数 (R²)

MSE_bpd2 = immse(output_test, double(BPoutput));  % 均方误差
RMSE_bpd2 = sqrt(MSE_bpd2);  % 均方根误差

% 打印结果
fprintf('R² = %.4f\n', R2_bpd2);
fprintf('MSE = %.4f\n', MSE_bpd2);
fprintf('RMSE = %.4f\n', RMSE_bpd2);
end