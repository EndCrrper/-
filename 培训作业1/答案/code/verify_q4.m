xm = [6.6857e-04,2.1313e+04,8.1830e-04,1.2193e+03,9.8977e-04,7.2606e+02,8.6450e-04,6.0728e+02,5.4317e-04,9.9975e+02];

% Q3 from Table 6 (canonical)
F_q3 = [25, 178.1, 194.3, 225.9, 265.0, 25];
[Tc3, ~, ~] = solve_heat(F_q3, 90.90/60, xm);
Ts3 = Tc3(Tc3 >= 30);
m3 = analyze_curve(Ts3);
s3 = calc_sym(Ts3);
fprintf('Q3_Table6: area=%.1f sigma=%.4f Tmax=%.1f slope=%.3f t150-190=%.1f t>217=%.1f\n', ...
    m3.area, s3, m3.Tmax, m3.max_slope, m3.t_150_190, m3.t_above);

% Q4 from current PSO run
F_q4 = [25, 176.4, 187.7, 230.6, 264.9, 25];
[Tc4, ~, ~] = solve_heat(F_q4, 90.28/60, xm);
Ts4 = Tc4(Tc4 >= 30);
m4 = analyze_curve(Ts4);
[s4, s1, s2, AL, AR] = calc_sym(Ts4);
fprintf('Q4: area=%.1f sigma=%.4f sigma1=%.4f sigma2=%.4f Tmax=%.1f slope=%.3f t150-190=%.1f t>217=%.1f\n', ...
    m4.area, s4, s1, s2, m4.Tmax, m4.max_slope, m4.t_150_190, m4.t_above);

% Q3 comparison from Q4 run (with actual re-optimized parameters)
F_q3c = [25, 182.7, 196.1, 231.2, 265.0, 25];
[Tc3c, ~, ~] = solve_heat(F_q3c, 95.12/60, xm);
Ts3c = Tc3c(Tc3c >= 30);
m3c = analyze_curve(Ts3c);
s3c = calc_sym(Ts3c);
fprintf('Q3_reopt: area=%.1f sigma=%.4f Tmax=%.1f slope=%.3f t150-190=%.1f t>217=%.1f\n', ...
    m3c.area, s3c, m3c.Tmax, m3c.max_slope, m3c.t_150_190, m3c.t_above);

fprintf('\nAll constraints ok: Q3=%d Q4=%d\n', check_constraints(m3), check_constraints(m4));
fprintf('Delta_area = %.1f (%.1f%%)\n', m4.area - m3.area, (m4.area - m3.area)/m3.area*100);
fprintf('Delta_sigma = %.4f (%.1f%%)\n', s4 - s3, (s4 - s3)/s3*100);

function [sigma, s1, s2, AL, AR] = calc_sym(T)
    dt = 0.5;
    above = T - 217;
    above = above(above >= 0);
    if isempty(above), sigma = 1; s1 = 1; s2 = 1; AL = 0; AR = 0; return; end
    [~, k] = max(above); n = length(above);
    if k > 1
        AL = (sum(above(1:k)) - (above(1)+above(k))/2) * dt;
    else
        AL = above(1)*dt/2;
    end
    if n-k > 1
        AR = (sum(above(k:n)) - (above(n)+above(k))/2) * dt;
    else
        AR = above(k)*dt/2;
    end
    s1 = abs(AL-AR)/max(abs(AL), abs(AR));
    s2 = abs(n+1-2*k)/max(k-1, n-k);
    sigma = max(s1, s2);
end
