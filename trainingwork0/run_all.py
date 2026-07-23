#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A题自来水厂水质预测与评估 - 一键运行全部求解脚本"""
import subprocess, sys, os, time

scripts = [
    ("数据预处理", "preprocess_data.py"),
    ("预处理可视化", "visualize_preprocess.py"),
    ("问题1：因素筛选与NTU预测", "solve_problem1.py"),
    ("问题2：滤后水动态时滞建模", "solve_problem2.py"),
    ("问题3：出厂水混合预测", "solve_problem3.py"),
    ("问题4：水质风险评价", "solve_problem4.py"),
]

print("="*60)
print("A题 自来水厂水质预测与评估 - 批量求解")
print("="*60)

start_all = time.time()
for name, script in scripts:
    print(f"\n{'='*60}")
    print(f">>> 运行: {name} ({script})")
    print("="*60)
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=os.path.dirname(__file__))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"[OK] {name} 完成 (耗时 {elapsed:.1f}s)")
        # Print last 10 lines
        lines = result.stdout.strip().split('\n')
        for line in lines[-10:]:
            print(f"  {line}")
    else:
        print(f"[FAIL] {name} 失败! (退出码: {result.returncode})")
        print(f"  STDOUT (last 20 lines):")
        for line in result.stdout.strip().split('\n')[-20:]:
            print(f"    {line}")
        print(f"  STDERR:")
        for line in result.stderr.strip().split('\n')[-10:]:
            print(f"    {line}")

total = time.time() - start_all
print(f"\n{'='*60}")
print(f"全部完成! 总耗时: {total:.1f}s")
print(f"输出目录: {os.path.join(os.path.dirname(__file__), 'output')}")
print("="*60)
