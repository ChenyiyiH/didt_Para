% 清空环境变量（根据需要）
% clc;
% clear;

% 读取数据
% (假设 input_test 和 error 已经被定义并加载)

% 提取数据
x_data = output_test(1, :); % x轴数据
y_data = output_test(2, :); % y轴数据
error = error_bpd2;

% 检查 error 的行数
if size(error, 1) < 2
    error('Error matrix does not have enough rows. Ensure error has at least 2 rows.');
end

% 提取 error 数据
error_data1 = error(1, :); % z轴数据1
error_data2 = error(2, :); % z轴数据2

% 创建网格坐标
num_samples = length(x_data); % 样本数

% 绘制散点图
% figure;

% subplot(1, 2, 1);
figure(1)
scatter3(x_data, y_data, error_data1, 10, error_data1, 'filled');
title('Error - R');
xlabel('R');
ylabel('C');
zlabel('Error Value');
colorbar; % 显示颜色条
colormap(jet); % 设置颜色映射
figure(2)
% subplot(1, 2, 2);
scatter3(x_data, y_data, error_data2, 10, error_data2, 'filled');
title('Error - C');
xlabel('R');
ylabel('C');
zlabel('Error Value');
colorbar; % 显示颜色条
colormap(jet); % 设置颜色映射
