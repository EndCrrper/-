# %% 自来水厂水质预测与评估 —— 一键运行全部求解脚本
"""
2020 CUMCM A题：自来水厂水质预测与评估
主运行脚本 —— 依次运行数据预处理和四个问题的求解
耗时约 2-3 分钟 (P1最耗时)
"""
import subprocess, sys, os, time

PYTHON = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ('数据预处理',       'preprocess.py'),
    ('问题1：因素筛选与预测',   'solve_problem1.py'),
    ('问题2：时滞动态建模',    'solve_problem2.py'),
    ('问题3：混合动态预测',    'solve_problem3.py'),
    ('问题4：水质风险评价',    'solve_problem4.py'),
]

print('=' * 60)
print('A题 自来水厂水质预测与评估 —— 批量求解')
print('=' * 60)

start_all = time.time()
for name, script in scripts:
    print(f'\n{"="*60}')
    print(f'>>> 运行: {name} ({script})')
    print('=' * 60)
    t0 = time.time()
    result = subprocess.run([PYTHON, script],
        capture_output=True, text=True, cwd=BASE_DIR)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f'[OK] {name} 完成 (耗时 {elapsed:.1f}s)')
        lines = result.stdout.strip().split('\n')
        for line in lines[-8:]:
            print(f'  {line}')
    else:
        print(f'[FAIL] {name} 失败! (退出码: {result.returncode})')
        print(f'  STDOUT:')
        for line in result.stdout.strip().split('\n')[-15:]:
            print(f'    {line}')
        print(f'  STDERR:')
        for line in result.stderr.strip().split('\n')[-5:]:
            print(f'    {line}')

total = time.time() - start_all
print(f'\n{"="*60}')
print(f'全部完成! 总耗时: {total:.1f}s')
print(f'输出目录: {os.path.join(BASE_DIR, "..", "result")}')
print('=' * 60)
