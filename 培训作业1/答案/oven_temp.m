function T_air = oven_temp(x, F)
% oven_temp - 计算回焊炉内位置 x 处的环境温度
%
% 输入:
%   x : 炉内位置 (cm)，可以是标量或向量
%   F : 各段设定温度 [T_front, T_zone1_5, T_zone6, T_zone7, T_zone8_9, T_rear]
%       其中 T_front = T_rear = 25 (车间温度)
%
% 输出:
%   T_air : 位置 x 处的炉内环境温度

    % 几何参数
    l = 5;      % 间隙长度 (cm)
    L = 30.5;   % 小温区长度 (cm)
    s = 25;     % 炉前/炉后区域长度 (cm)

    % 各段分界点位置
    x1 = 0;          % 起点
    x2 = s;          % 炉前结束 = 25
    x3 = x2 + 5*L + 4*l;   % 温区5结束 = 25 + 152.5 + 20 = 197.5
    x4 = x3 + l;            % 间隙结束 = 202.5
    x5 = x4 + L;            % 温区6结束 = 233
    x6 = x5 + l;            % 间隙结束 = 238
    x7 = x6 + L;            % 温区7结束 = 268.5
    x8 = x7 + l;            % 间隙结束 = 273.5
    x9 = x8 + 2*L + l;      % 温区9结束 = 273.5 + 61 + 5 = 339.5
    x10 = x9 + l;           % 间隙结束 = 344.5

    % 分段线性函数计算
    T_air = zeros(size(x));
    for i = 1:length(x)
        xi = x(i);
        if xi <= x2
            % 炉前区域：线性从 F(1)=25 升至 F(2)
            T_air(i) = (F(2) - F(1)) / s * (xi - x1) + F(1);
        elseif xi <= x3
            % 温区1~5：恒温 F(2)
            T_air(i) = F(2);
        elseif xi <= x4
            % 间隙：线性从 F(2) 过渡到 F(3)
            T_air(i) = (F(3) - F(2)) / l * (xi - x3) + F(2);
        elseif xi <= x5
            % 温区6：恒温 F(3)
            T_air(i) = F(3);
        elseif xi <= x6
            % 间隙：线性从 F(3) 过渡到 F(4)
            T_air(i) = (F(4) - F(3)) / l * (xi - x5) + F(3);
        elseif xi <= x7
            % 温区7：恒温 F(4)
            T_air(i) = F(4);
        elseif xi <= x8
            % 间隙：线性从 F(4) 过渡到 F(5)
            T_air(i) = (F(5) - F(4)) / l * (xi - x7) + F(4);
        elseif xi <= x9
            % 温区8~9：恒温 F(5)
            T_air(i) = F(5);
        elseif xi <= x10
            % 间隙：线性从 F(5) 过渡到 F(6)=25
            T_air(i) = (F(6) - F(5)) / l * (xi - x9) + F(5);
        else
            % 炉后区域：恒温 F(6)=25
            T_air(i) = F(6);
        end
    end
end
