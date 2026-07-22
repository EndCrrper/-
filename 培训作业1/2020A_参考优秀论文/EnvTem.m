function y = EnvTem(x)
    % 预定义温度值和参数
    F = [25, 175, 195, 235, 255, 25];
    l = 5; L = 30.5; s = 25;
    
    % 计算分界点
    x1 = 0;
    x2 = s;
    x3 = x2 + 5*L + 4*l;
    x4 = x3 + l;
    x5 = x4 + L;
    x6 = x5 + l;
    x7 = x6 + L;
    x8 = x7 + l;
    x9 = x8 + 2*L + l;
    x10 = x9 + l;
    x11 = x10 + 2*L + l + s;

    % 定义所有区间的分界点
    edges = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11];
    
    % 使用 discretize 确定每个 x 所属的区间索引
    [~, ~, indices] = histcounts(x, edges);
    
    % 初始化输出数组
    y = zeros(size(x));
    
    % 处理每个区间的计算
    for i = 1:10
        mask = (indices == i);
        if any(mask)
            switch i
                case 1  % 0 <= x <= x2: 线性上升
                    y(mask) = (F(2)-F(1))/s * (x(mask) - 0) + F(1);
                case 2  % x2 < x <= x3: 恒定
                    y(mask) = F(2);
                case 3  % x3 < x <= x4: 线性上升
                    y(mask) = (F(3)-F(2))/l * (x(mask) - x3) + F(2);
                case 4  % x4 < x <= x5: 恒定
                    y(mask) = F(3);
                case 5  % x5 < x <= x6: 线性上升
                    y(mask) = (F(4)-F(3))/l * (x(mask) - x5) + F(3);
                case 6  % x6 < x <= x7: 恒定
                    y(mask) = F(4);
                case 7  % x7 < x <= x8: 线性上升
                    y(mask) = (F(5)-F(4))/l * (x(mask) - x7) + F(4);
                case 8  % x8 < x <= x9: 恒定
                    y(mask) = F(5);
                case 9  % x9 < x <= x10: 线性下降
                    y(mask) = (F(6)-F(5))/l * (x(mask) - x9) + F(5);
                case 10 % x > x10: 恒定
                    y(mask) = F(6);
            end
        end
    end
end