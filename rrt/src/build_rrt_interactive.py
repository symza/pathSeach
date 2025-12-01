#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RRT交互式演示程序打包脚本
将 rrt_interactive.py 打包为单个可执行文件
"""
import os
import sys
import subprocess
import shutil

# 获取脚本所在目录和rrt文件夹路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src文件夹
RRT_DIR = os.path.dirname(SCRIPT_DIR)  # rrt文件夹

def check_pyinstaller():
    """检查并安装PyInstaller"""
    try:
        import PyInstaller
        print("[✓] PyInstaller 已安装")
        return True
    except ImportError:
        print("[!] PyInstaller 未安装，正在安装...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("[✓] PyInstaller 安装成功")
            return True
        except subprocess.CalledProcessError:
            print("[✗] PyInstaller 安装失败！")
            return False

def clean_old_files():
    """清理旧的打包文件"""
    print("\n[步骤 2/3] 清理旧文件...")
    
    # 清理src和rrt文件夹中的build和dist目录
    dirs_to_remove = ['build', 'dist']
    for base_dir in [SCRIPT_DIR, RRT_DIR]:
        for dir_name in dirs_to_remove:
            dir_path = os.path.join(base_dir, dir_name)
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    print(f"  [✓] 已删除目录: {dir_path}")
                except Exception as e:
                    print(f"  [!] 删除目录 {dir_path} 失败: {e}")
    
    # 删除spec文件（在src和rrt文件夹中）
    for base_dir in [SCRIPT_DIR, RRT_DIR]:
        spec_file = os.path.join(base_dir, 'RRTInteractive.spec')
        if os.path.exists(spec_file):
            try:
                os.remove(spec_file)
                print(f"  [✓] 已删除文件: {spec_file}")
            except Exception as e:
                print(f"  [!] 删除文件 {spec_file} 失败: {e}")

def build_exe(debug=False):
    """打包为可执行文件
    
    参数:
        debug: 如果为True，创建带控制台窗口的调试版本
    """
    print("\n[步骤 3/3] 开始打包...")
    if debug:
        print("[调试模式] 将创建带控制台窗口的版本以便查看错误信息")
    print("[提示] 这个过程可能需要几分钟，请耐心等待...\n")
    
    # 源文件在src文件夹中
    source_file = os.path.join(SCRIPT_DIR, "rrt_interactive.py")
    if not os.path.exists(source_file):
        print(f"[✗] 错误: 找不到源文件 {source_file}")
        return False
    
    # 切换到src目录执行打包（PyInstaller会在当前目录创建build和dist）
    original_cwd = os.getcwd()
    try:
        os.chdir(SCRIPT_DIR)
        
        # 确定输出文件名
        exe_name = "RRTInteractive_Debug" if debug else "RRTInteractive"
        
        # PyInstaller 命令参数
        # --onefile: 打包成单个文件
        # --windowed: Windows下不显示控制台窗口（对于matplotlib图形界面）
        #   如果debug=True，则不使用--windowed，显示控制台窗口
        # --name: 指定输出文件名
        # --clean: 清理临时文件
        # --distpath: 指定dist目录位置（rrt文件夹）
        # --workpath: 指定build目录位置（rrt文件夹）
        # --hidden-import: 确保所有必要的模块都被包含
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--name", exe_name,
            "--clean",
            "--distpath", RRT_DIR,
            "--workpath", os.path.join(RRT_DIR, "build"),
            "--hidden-import", "matplotlib.backends.backend_tkagg",
            "--hidden-import", "matplotlib.backends.backend_qt5agg",
            "--hidden-import", "tkinter",
            "--hidden-import", "PIL._tkinter_finder",
        ]
        
        # 如果不是调试模式，添加--windowed参数
        if not debug:
            cmd.append("--windowed")
        
        cmd.append("rrt_interactive.py")
        
        subprocess.check_call(cmd)
        
        # 移动exe文件到rrt文件夹（如果不在的话）
        exe_src = os.path.join(RRT_DIR, f"{exe_name}.exe")
        exe_dst = os.path.join(RRT_DIR, f"{exe_name}.exe")
        
        if os.path.exists(exe_src) and exe_src != exe_dst:
            if os.path.exists(exe_dst):
                os.remove(exe_dst)
            shutil.move(exe_src, exe_dst)
        
        print("\n" + "="*50)
        print("打包完成！")
        print("="*50)
        print(f"\n[成功] 可执行文件位置: {exe_dst}")
        if debug:
            print("\n[调试版本] 此版本会显示控制台窗口，可以看到错误信息")
        print("\n[提示]")
        print("- 可以在任何 Windows 电脑上运行，无需安装 Python")
        print("- 源文件不会被删除")
        print("- 文件大小约 50-100MB（包含所有依赖）")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[✗] 打包失败！错误代码: {e.returncode}")
        print("请检查上方的错误信息。")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    """主函数"""
    print("="*50)
    print("RRT交互式演示程序 - 打包脚本")
    print("="*50)
    
    # 询问是否创建调试版本
    print("\n[选项] 是否创建调试版本？")
    print("  - 调试版本：显示控制台窗口，可以看到错误信息（推荐用于首次打包）")
    print("  - 正式版本：不显示控制台窗口，界面更简洁")
    debug_choice = input("\n创建调试版本？(y/n，默认n): ").strip().lower()
    debug_mode = debug_choice == 'y' or debug_choice == 'yes'
    
    # 步骤1: 检查PyInstaller
    print("\n[步骤 1/3] 检查 PyInstaller...")
    if not check_pyinstaller():
        input("\n按回车键退出...")
        sys.exit(1)
    
    # 步骤2: 清理旧文件
    clean_old_files()
    
    # 步骤3: 打包
    if build_exe(debug=debug_mode):
        print("\n" + "="*50)
        input("\n按回车键退出...")
    else:
        input("\n按回车键退出...")
        sys.exit(1)

if __name__ == "__main__":
    main()

