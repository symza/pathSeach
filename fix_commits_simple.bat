@echo off
chcp 65001 >nul
echo ================================================
echo 修复 Git 提交信息中文乱码（简化版）
echo ================================================
echo.
echo 此脚本将修复最近的 3 个提交信息
echo.
echo 警告：需要强制推送到远程仓库
echo.
pause

echo.
echo [步骤 1/3] 设置 Git 编码...
git config --local i18n.commitencoding utf-8
git config --local i18n.logoutputencoding utf-8
git config --local core.quotepath false

echo.
echo [步骤 2/3] 开始交互式 rebase...
echo 请在编辑器中：
echo 1. 将需要修改的提交前的 pick 改为 reword
echo 2. 保存并关闭
echo 3. 逐个修改提交信息为正确的中文
echo.
pause

git rebase -i HEAD~4

echo.
echo [步骤 3/3] 如果修改成功，运行以下命令强制推送：
echo git push -f origin master
echo.
pause

