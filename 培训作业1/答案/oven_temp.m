function T_air = oven_temp(x, F)
% 炉内环境温度分布 (分段线性)
% F = [T_front, T1_5, T6, T7, T8_9, T_rear]

    l = 5; L = 30.5; s = 25;

    x1 = 0;          x2 = s;                    % 25
    x3 = x2 + 5*L + 4*l;   x4 = x3 + l;         % 197.5, 202.5
    x5 = x4 + L;            x6 = x5 + l;         % 233, 238
    x7 = x6 + L;            x8 = x7 + l;         % 268.5, 273.5
    x9 = x8 + 2*L + l;      x10 = x9 + l;        % 339.5, 344.5

    for i = 1:length(x)
        xi = x(i);
        if xi <= x2
            T_air(i) = (F(2)-F(1))/s*(xi-x1) + F(1);
        elseif xi <= x3
            T_air(i) = F(2);
        elseif xi <= x4
            T_air(i) = (F(3)-F(2))/l*(xi-x3) + F(2);
        elseif xi <= x5
            T_air(i) = F(3);
        elseif xi <= x6
            T_air(i) = (F(4)-F(3))/l*(xi-x5) + F(3);
        elseif xi <= x7
            T_air(i) = F(4);
        elseif xi <= x8
            T_air(i) = (F(5)-F(4))/l*(xi-x7) + F(4);
        elseif xi <= x9
            T_air(i) = F(5);
        elseif xi <= x10
            T_air(i) = (F(6)-F(5))/l*(xi-x9) + F(5);
        else
            T_air(i) = F(6);
        end
    end
end
