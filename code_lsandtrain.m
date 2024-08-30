%% 含有电容杂散电感与电阻杂散电感的数据集创建
clc
clear
%% 定义系统初值
Ud = 2400;      % 直流电压
Lc = 4e-7;      % 母排杂散电感
Lr = 3e-7;      % 电阻杂散电感
L = 4e-6;       % 缓冲回路电抗
C = 34e-6;      % 缓冲回路电容
R = 0.45;       % 缓冲回路电阻
It = 4000;      % 输出电流

is_Ls = 0;      % 计算杂散电感影响
choose_lr = 0;  % 是否含有Lr，与is_Ls相关

is_data = 0;    % 构造数据
choose = 1;     % 是否构造数据集，与is_data相关

is_train = 0;   % 训练

is_getfun = 0;  % 获取模型数据，训练后

is_varify = 1;  % 验证数据，训练后
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1：双隐层BP1 √         2：双隐层BP2  √     3：遗传算法优化BP  √          %
% 4：粒子群算法优化BP √  5：单隐层BP1  √     6：单隐层BP2   √              %
% 7：极限学习机ELM       8：ELMAN神经网络    9：GRNN        √              %
% 10：LSTM               11：SCNN            12：RBF（可选RBE）            %
% 13：PNN                14：SVM             15：小波神经网络（波函数可选） %
% 16: 单隐层BP（稳定版）                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
chose_model = 16;                % 选择模型1-16 % SCNN不太行 PNN一般 wavenn不行
train_par = 0.8;                 % 训练集比例

dataname = sprintf("data_%uuH.mat", L*1e6);             % 数据文件
% load(dataname,'in','out');

%% 计算杂散电感的影响（[Lsc,Lsr]）
if is_Ls
    if choose_lr
        for i = 1:10
            Lc = 1e-7;
            for j = 1:10
                % 定义微分方程
                dydt_3 = @(t, Y) [Y(2);
                    Y(3);
                    -(((L*R*C+Lc*C*R)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(3) +...
                    ((L+Lr)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(2) + ...
                    (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(1) - (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Ud)];
                % 定义初值
                Y0 = [Ud; It / C; 0];
                % 解决常微分方程
                [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);
                [m,idx]=max(Y_3(:,1));
                % plot(t_3,Y_3(:, 1));
                % hold on
                tt = t_3(idx);
                Um(i,j) = m;
                tm(i,j) = tt;
                Lc = Lc+1e-7;
            end
            Lr = Lr+1e-7;
        end

        %% 绘图
        % 光滑度查看
        lr = 1e-7:1e-7:1.0e-6;
        lc = 1e-7:1e-7:1.0e-6;
        [lc,lr] = meshgrid(lc,lr);
        figure(5)
        surf(lc,lr,Um);
        title('UM');
        xlabel('lc');
        ylabel('lr');
        zlabel('Um');
        figure(6)
        surf(lr,lc,tm);
        title('tM');
        xlabel('lc');
        ylabel('lr');
        zlabel('tm');

    elseif choose_lr == 0

        Lc = 1e-7;
        for i = 1:10
            % 定义微分方程
            dydt_3 = @(t, Y) [Y(2);
                Y(3);
                -((L + Lc) * R / (L * Lc) * Y(3) + (1 / (Lc * C)) * Y(2) + (R / (Lc * L * C)) * Y(1) - R * Ud / (Lc * L * C))];
            % 定义初值
            Y0 = [Ud; It / C; 0];
            % 解决常微分方程
            [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);
            [m,idx]=max(Y_3(:,1));
            % plot(t_3,Y_3(:, 1));
            % hold on
            tt = t_3(idx);
            Um(i) = m;
            tm(i) = tt;
            Lc = Lc+1e-7;
        end
        lc = 1e-7:1e-7:1.0e-6;
        plot3(lc,Um,tm)
        xlabel('lc');
        ylabel('um');
        zlabel('tm');
    end
end
%% 数据集构造
if(is_data)
    if(choose == 0)
        %% 单个It（看光滑度）

        C_initial = 10e-6;
        C_step = 1e-6;
        C_end = 49e-6;
        R_initial = 0.3;
        R_step = 0.01;
        R_end = 0.89;

        % 计算C和R的范围
        C_values = C_initial:C_step:C_end;
        R_values = R_initial:R_step:R_end;
        num_C = length(C_values);
        num_R = length(R_values);

        % 预分配结果矩阵
        Um = zeros(num_R, num_C);
        tm = zeros(num_R, num_C);

        % 主循环计算
        for j = 1:num_R
            R = R_values(j);
            for l = 1:num_C
                C = C_values(l);

                % 定义微分方程
                dydt_3 = @(t, Y) [Y(2);
                    Y(3);
                    -((L + Lc) * R / (L * Lc) * Y(3) ...
                    + (1 / (Lc * C)) * Y(2) ...
                    + (R / (Lc * L * C)) * Y(1)...
                    - R * Ud / (Lc * L * C))];

                % 定义初值
                Y0 = [Ud; It / C; 0];

                % 解决常微分方程
                [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);

                % 查找最大值及其对应时间
                [m, idx] = max(Y_3(:,1));
                tt = t_3(idx);

                % 存储结果
                Um(j, l) = m - Ud;
                tm(j, l) = tt;
            end
        end

        % 绘制结果
        [R_grid, C_grid] = meshgrid(R_values, C_values);

        figure('Color',[1 1 1]);
        surf(R_grid, C_grid, Um');
        title('UM');
        xlabel('R');
        ylabel('C');
        zlabel('Um');

        figure('Color',[1 1 1]);
        surf(R_grid, C_grid, tm');
        title('TM');
        xlabel('R');
        ylabel('C');
        zlabel('tm');

    elseif(choose==1)
        %% 构造数据集
        % 计算循环次数
        num_C = 40;
        num_R = 50;
        % 初始化参数
        C_initial = 10e-6;
        C_step = 1e-6;
        C_end = C_initial + (num_C - 1) * C_step; % num_C步，最后一个步长
        R_initial = 0.2;
        R_step = 0.01;
        R_end = R_initial + (num_R - 1) * R_step;

        rr = R_initial:R_step:R_end;
        cc = C_initial:C_step:C_end;
        [C_grid,R_grid] = meshgrid(cc,rr);
        % 预分配结果矩阵
        n_ori = zeros(num_C * num_R, 3);
        Um = zeros(num_C * num_R, 1);
        tm = zeros(num_C * num_R, 1);

        % 初始化计数器
        cnt = 1;

        % 主循环计算
        for j = 1:num_R
            R = R_initial + (j - 1) * R_step; % 更新R
            C = C_initial; % 每次内层循环开始时重置C
            for l = 1:num_C
                % 存储参数
                n_ori(cnt, :) = [R, C, It];

                % 定义微分方程
                % dydt_3 = @(t, Y) [Y(2);
                %     Y(3);
                %     -(((L*R*C+Lc*C*R)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(3) +...
                %     ((L+Lr)/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(2) + ...
                %     (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Y(1) - (R/(L*Lr*C+L*Lc*C+Lc*Lr*C)) * Ud)];

                % 定义微分方程
                dydt_3 = @(t, Y) [Y(2);
                    Y(3);
                    -((L + Lc) * R / (L * Lc) * Y(3) ...
                    + (1 / (Lc * C)) * Y(2) ...
                    + (R / (Lc * L * C)) * Y(1)...
                    - R * Ud / (Lc * L * C))];

                % 定义初值
                Y0 = [Ud; It / C; 0];

                % 解决常微分方程
                [t_3, Y_3] = ode45(dydt_3, [0 1e-4], Y0);

                % 找到最大值及其对应时间
                [m, idx] = max(Y_3(:, 1));
                tt = t_3(idx);

                % 存储结果
                Um(cnt) = m - Ud;
                tm(cnt) = tt;

                % 更新C
                C = C + C_step;
                cnt = cnt + 1;
            end
        end

        % 保存数据
        data.out = n_ori(:, 1:2);
        data.in = [Um, tm];
        in = data.in;
        out = data.out;
        savename = dataname;
        save(savename, 'data', 'in', 'out','R_grid','C_grid');

    end
end

%% 训练模型
if is_train

    model_functions = {@BP_double, @BP_double2, @BP_GA, @BP_PSO, @BP_single, ...
        @BP_single2, @ELM, @ELMAN, @GRNN, @LSTM, @SCNN, ...
        @RBF, @PNN, @SVM, @wavenn,@BP_single1};
    if chose_model >= 1 && chose_model <= numel(model_functions)
        [~,~,~,~,net] = model_functions{chose_model}(dataname,train_par);
        % disp(results);
    end
end

%% 生成预测文件
if is_getfun
    genFunction(net,'predict.m','MatrixOnly','yes');
    % mcc -W lib:libpredict -T link:lib predict
end
%% 验证
if is_varify
    load(dataname);
    out_i = predict(in');
    out_i = out_i';
    error_i = (out-out_i)./out_i;
    error_1 = reshape(error_i(:,1),[50 40]);
    error_2 = reshape(error_i(:,2),[50 40]);
    figure('Color',[1 1 1]);
    surf(C_grid, R_grid, error_1);
    colorbar;
    figure('Color',[1 1 1]);
    surf(C_grid, R_grid, error_2);
    colorbar;
    idx_rand = randi([1, 2000], 1, 1);
    sim('test_var.slx');
end
