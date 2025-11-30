@echo off
chcp 65001 >nul
echo ================================================
echo Git 中文编码配置脚本
echo ================================================
echo.

echo [步骤 1/3] 配置 Git 使用 UTF-8 编码...
git config --global core.quotepath false
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
git config --global gui.encoding utf-8

echo.
echo [步骤 2/3] 配置本地仓库编码...
git config --local core.quotepath false
git config --local i18n.commitencoding utf-8
git config --local i18n.logoutputencoding utf-8

echo.
echo [步骤 3/3] 设置控制台编码为 UTF-8...
chcp 65001

echo.
echo ================================================
echo 配置完成！
echo ================================================
echo.
echo 提示：
echo 1. 当前控制台已设置为 UTF-8 编码
echo 2. 如果仍有乱码，请关闭并重新打开命令行窗口
echo 3. 或使用 Git GUI 工具查看提交信息
echo.
echo 验证配置：
git config --list | findstr encoding
echo.
pause

