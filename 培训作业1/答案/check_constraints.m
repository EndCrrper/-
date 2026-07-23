function ok = check_constraints(metrics)
% check_constraints - 检查炉温曲线是否满足制程界限
%
% 制程界限:
%   1. 240 <= Tmax <= 250 °C
%   2. 40 <= t_above <= 90 s  (温度>217°C的时间)
%   3. max_slope <= 3 °C/s
%   4. 60 <= t_150_190 <= 120 s  (升温过程150~190°C时间)

    ok = (metrics.Tmax >= 240) && (metrics.Tmax <= 250) && ...
         (metrics.t_above >= 40) && (metrics.t_above <= 90) && ...
         (metrics.max_slope <= 3.0) && ...
         (metrics.t_150_190 >= 60) && (metrics.t_150_190 <= 120);
end
