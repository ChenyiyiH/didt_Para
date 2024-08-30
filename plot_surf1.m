% 清空环境变量（根据需要）
% clc;
% clear;

% 假设 input_test 和 error 已经被定义并加载

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

% 创建均匀网格
[x_grid, y_grid] = meshgrid(linspace(min(x_data), max(x_data), sqrt(num_samples)), ...
                            linspace(min(y_data), max(y_data), sqrt(num_samples)));

% 将 1D error 数据映射到网格
% 使用 griddata 进行插值以适应网格
Z1 = griddata(x_data, y_data, error_data1, x_grid, y_grid, 'linear');
Z2 = griddata(x_data, y_data, error_data2, x_grid, y_grid, 'linear');

% 绘制 surf 图
% figure;
figure(1)
% subplot(1, 2, 1);
surf(x_grid, y_grid, Z1);
title('Error - R');
xlabel('R');
ylabel('C');
zlabel('Error Value');
colorbar; % 显示颜色条
colormap(jet); % 设置颜色映射
figure(2)
% subplot(1, 2, 2);
surf(x_grid, y_grid, Z2);
title('Error - C');
xlabel('R');
ylabel('C');
zlabel('Error Value');
colorbar; % 显示颜色条
colormap(jet); % 设置颜色映射
