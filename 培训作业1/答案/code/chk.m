function chk()
    xm = [6.6857e-04,2.1313e+04,8.1830e-04,1.2193e+03,9.8977e-04,7.2606e+02,8.6450e-04,6.0728e+02,5.4317e-04,9.9975e+02];
    fid = fopen('chk_out.txt', 'w');

    F3 = [25, 178.1, 194.3, 225.9, 265.0, 25];
    [Tc, ~, ~] = solve_heat(F3, 90.90/60, xm);
    Ts = Tc(Tc >= 30);
    m = analyze_curve(Ts);
    fprintf(fid, 'Q3: Tmax=%.1f slope=%.3f t150-190=%.1f t>217=%.1f area=%.1f ok=%d\n', m.Tmax, m.max_slope, m.t_150_190, m.t_above, m.area, check_constraints(m));

    F4 = [25, 176.4, 187.7, 230.6, 264.9, 25];
    [Tc, ~, ~] = solve_heat(F4, 90.28/60, xm);
    Ts = Tc(Tc >= 30);
    m = analyze_curve(Ts);
    fprintf(fid, 'Q4: Tmax=%.1f slope=%.3f t150-190=%.1f t>217=%.1f area=%.1f ok=%d\n', m.Tmax, m.max_slope, m.t_150_190, m.t_above, m.area, check_constraints(m));

    fclose(fid);
    fprintf('done\n');
end
