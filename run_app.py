# -*- coding: utf-8 -*-
# run_app.py - 启动 Web 界面的脚本

import os
import sys
import subprocess

def main():
    print("=" * 50)
    print("   Qwen Web 界面启动器")
    print("=" * 50)
    print()
    
    # 检查 gradio 是否安装
    try:
        import gradio
        print("[OK] Gradio 已安装")
    except ImportError:
        print("[安装中] 正在安装 Gradio...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio", "-q"])
        print("[OK] Gradio 安装完成")
    
    print()
    print("启动 Web 服务...")
    print("访问地址: http://127.0.0.1:7860")
    print()
    print("按 Ctrl+C 停止服务")
    print("-" * 50)
    
    # 启动 app.py
    os.system("python app.py")

if __name__ == "__main__":
    main()