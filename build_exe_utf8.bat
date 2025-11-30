@echo off
REM 路径搜索算法演示程序 - 打包脚本
REM 使用 UTF-8 编码保存此文件
chcp 65001 >nul
echo ================================================
echo 路径搜索算法演示程序 - 打包脚本
echo ================================================
echo.

echo [步骤 1/3] 检查 PyInstaller...
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [信息] 正在安装 PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo [错误] PyInstaller 安装失败！
        pause
        exit /b 1
    )
)

echo.
echo [步骤 2/3] 清理旧文件...
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist *.spec del /q *.spec 2>nul

echo.
echo [步骤 3/3] 开始打包...
echo [提示] 这个过程可能需要几分钟，请耐心等待...
echo.

python -m PyInstaller --windowed --onefile --name PathfindingMazeDemo pathfinding_maze_app.py

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！
    echo 请检查上方的错误信息。
    pause
    exit /b 1
)

echo.
echo ================================================
echo 打包完成！
echo ================================================
echo.
echo [成功] 可执行文件位置: dist\PathfindingMazeDemo.exe
echo.
echo [提示]
echo - 可以在任何 Windows 电脑上运行，无需安装 Python
echo - 源文件 pathfinding_maze_app.py 不会被删除
echo - 文件大小约 50-100MB（包含所有依赖）
echo.
pause

