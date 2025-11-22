#!/usr/bin/env python3
"""
Web系统测试脚本
快速验证系统是否正常工作
"""

import sys
from pathlib import Path

print("="*50)
print("Web系统环境检查")
print("="*50)

# 1. 检查Python版本
print("\n1. Python版本检查...")
print(f"   ✓ Python {sys.version.split()[0]}")

# 2. 检查依赖
print("\n2. 依赖包检查...")
try:
    import flask
    print(f"   ✓ Flask {flask.__version__}")
except ImportError:
    print("   ✗ Flask 未安装")
    print("   请运行: pip3 install flask")
    sys.exit(1)

try:
    import pandas
    print(f"   ✓ Pandas {pandas.__version__}")
except ImportError:
    print("   ✗ Pandas 未安装")
    sys.exit(1)

# 3. 检查文件结构
print("\n3. 文件结构检查...")
required_files = [
    "web_app.py",
    "templates/index.html",
    "static/css/style.css",
    "static/js/app.js",
    "configs.json",
    "Selector.py",
    "select_stock.py"
]

missing_files = []
for file in required_files:
    path = Path(file)
    if path.exists():
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} (缺失)")
        missing_files.append(file)

if missing_files:
    print(f"\n   警告: {len(missing_files)} 个文件缺失")
    sys.exit(1)

# 4. 检查数据目录
print("\n4. 数据目录检查...")
data_dir = Path("./data")
if not data_dir.exists():
    print("   ✗ data 目录不存在")
    print("   请先下载数据: python3 fetch_kline.py")
    sys.exit(1)

csv_files = list(data_dir.glob("*.csv"))
print(f"   ✓ 数据目录存在")
print(f"   ✓ 股票数据: {len(csv_files)} 个文件")

if len(csv_files) == 0:
    print("   警告: 数据文件为空，请先下载数据")

# 5. 尝试导入模块
print("\n5. 模块导入检查...")
try:
    from select_stock import load_data, load_config
    print("   ✓ select_stock 模块")
except Exception as e:
    print(f"   ✗ select_stock 导入失败: {e}")
    sys.exit(1)

try:
    from Selector import BreakoutPreviousHighSelector
    print("   ✓ Selector 模块")
except Exception as e:
    print(f"   ✗ Selector 导入失败: {e}")
    sys.exit(1)

# 6. 检查配置文件
print("\n6. 配置文件检查...")
try:
    selectors = load_config(Path("./configs.json"))
    active_count = sum(1 for s in selectors if s.get("activate", False))
    print(f"   ✓ 配置文件有效")
    print(f"   ✓ 战法总数: {len(selectors)}")
    print(f"   ✓ 已激活: {active_count}")
    
    # 显示战法列表
    print("\n   已激活的战法:")
    for s in selectors:
        if s.get("activate", False):
            print(f"     - {s.get('alias', s['class'])}")
except Exception as e:
    print(f"   ✗ 配置文件错误: {e}")
    sys.exit(1)

# 总结
print("\n" + "="*50)
print("✅ 所有检查通过！系统可以正常启动")
print("="*50)
print("\n启动命令:")
print("  ./start_web.sh")
print("或")
print("  python3 web_app.py")
print("\n访问地址:")
print("  http://127.0.0.1:5000")
print("\n" + "="*50)

