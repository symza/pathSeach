@echo off
chcp 65001 >nul
echo ================================================
echo 修复 Git 提交信息中文乱码
echo ================================================
echo.
echo 警告：此操作将修改提交历史，需要强制推送
echo 如果仓库是共享的，请谨慎操作
echo.
pause

echo.
echo [步骤 1/3] 配置 Git 编码...
git config --local i18n.commitencoding utf-8
git config --local i18n.logoutputencoding utf-8

echo.
echo [步骤 2/3] 开始修复提交信息...
echo.

REM 使用交互式 rebase 修复最近的提交
git rebase -i HEAD~4

echo.
echo [步骤 3/3] 如果修改成功，需要强制推送
echo 运行: git push -f origin master
echo.
pause

