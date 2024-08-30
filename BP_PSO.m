%% 该代码为基于PSO和BP网络的预测
function [error_bppso,R2_bppso,MSE_bppso,RMSE_bppso,net_psobp]=BP_PSO(datatable,train_par)
%% 清空环境
% clc;
% clear;
% close all;
nntwarn off;
load(datatable);
% 读取数据

input = data.in;
output = data.out;
k = rand(1,size(input,1));
[m, n] = sort(k);
n_idx = size(input,1) * train_par;
% 节点个数
inputnum = size(input,2);
hiddennum = 15;
outputnum = size(output,2);

% 训练数据和预测数据
input_train = input(n(1:n_idx), :)';
output_train = output(n(1:n_idx), :)';
input_test = input(n(n_idx+1:end),:)';
output_test = output(n(n_idx+1:end),:)';

% 样本输入输出数据归一化
[inputn, inputps] = mapminmax(input_train);
[outputn, outputps] = mapminmax(output_train);

% 构建网络
net_psobp = newff(inputn, outputn, hiddennum);

% 参数初始化
c1 = 1.49445; % 个体学习因子
c2 = 1.49445; % 群体学习因子
maxgen = 5; % 进化次数
sizepop = 200; % 种群规模
Vmax = 1;
Vmin = -1;
popmax = 5;
popmin = -5;

% 确定总参数数量
total_params = inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum;

% 初始化粒子群
pop = zeros(sizepop, total_params);
V = zeros(sizepop, total_params);
fitness = zeros(1, sizepop);

for i = 1:sizepop
    pop(i,:) = (popmax - popmin) * rand(1, total_params) + popmin;
    V(i,:) = (Vmax - Vmin) * rand(1, total_params) + Vmin;
    fitness(i) = fun(pop(i,:), inputnum, hiddennum, outputnum, net_psobp, inputn, outputn);
end

% 个体极值和群体极值
[bestfitness, bestindex] = min(fitness);
zbest = pop(bestindex, :); % 全局最佳
gbest = pop; % 个体最佳
fitnessgbest = fitness; % 个体最佳适应度值
fitnesszbest = bestfitness; % 全局最佳适应度值

%% 迭代寻优
for i = 1:maxgen
    disp(['Iteration: ', num2str(i)]);
    
    for j = 1:sizepop
        % 速度更新
        V(j,:) = V(j,:) + c1 * rand * (gbest(j,:) - pop(j,:)) + c2 * rand * (zbest - pop(j,:));
        V(j, V(j,:) > Vmax) = Vmax;
        V(j, V(j,:) < Vmin) = Vmin;
        
        % 种群更新
        pop(j,:) = pop(j,:) + 0.2 * V(j,:);
        pop(j, pop(j,:) > popmax) = popmax;
        pop(j, pop(j,:) < popmin) = popmin;
        
        % 自适应变异
        pos = unidrnd(total_params);
        if rand > 0.95
            pop(j, pos) = (popmax - popmin) * rand + popmin;
        end
        
        % 适应度值
        fitness(j) = fun(pop(j,:), inputnum, hiddennum, outputnum, net_psobp, inputn, outputn);
    end
    
    % 个体最优更新
    for j = 1:sizepop
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
    end
    
    % 群体最优更新
    [newbestfitness, newbestindex] = min(fitness);
    if newbestfitness < fitnesszbest
        zbest = pop(newbestindex,:);
        fitnesszbest = newbestfitness;
    end
    
    yy(i) = fitnesszbest;    
end

%% 结果分析
plot(yy);
title(['适应度曲线  ' '终止代数＝' num2str(maxgen)]);
xlabel('进化代数'); ylabel('适应度');

x = zbest;

%% 把最优初始阀值权值赋予网络预测
% 用粒子群优化的BP网络进行值预测
w1 = x(1:inputnum * hiddennum);
B1 = x(inputnum * hiddennum + 1:inputnum * hiddennum + hiddennum);
w2 = x(inputnum * hiddennum + hiddennum + 1:inputnum * hiddennum + hiddennum + hiddennum * outputnum);
B2 = x(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1:end);

net_psobp.iw{1, 1} = reshape(w1, hiddennum, inputnum);
net_psobp.lw{2, 1} = reshape(w2, outputnum, hiddennum);
net_psobp.b{1} = reshape(B1, hiddennum, 1);
net_psobp.b{2} = reshape(B2, outputnum, 1);
%% BP网络训练
%网络进化参数
net_psobp.trainParam.epochs=1000;
net_psobp.trainParam.lr=0.1;
%net.trainParam.goal=0.00001;

%网络训练
[net_psobp,per2]=train(net_psobp,inputn,outputn);

%% BP网络预测
%数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net_psobp,inputn_test);
test_simu=mapminmax('reverse',an,outputps);
error_bppso=test_simu-output_test;

%% 评估模型性能
% 决定系数 (R²)
R_bppso = corrcoef(output_test, test_simu);
R2_bppso = R_bppso(1, 2)^2;

% 均方误差和均方根误差
MSE_bppso = immse(output_test, double(test_simu));
RMSE_bppso = sqrt(MSE_bppso);

% 打印结果
fprintf('R² = %.4f\n', R2_bppso);
fprintf('MSE = %.4f\n', MSE_bppso);
fprintf('RMSE = %.4f\n', RMSE_bppso);
end