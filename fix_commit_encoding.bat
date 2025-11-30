@echo off
chcp 65001 >nul
echo ================================================
echo Fix Git Commit Message Encoding Issues
echo ================================================
echo.

REM 设置 Git 编码
git config --local i18n.commitencoding utf-8
git config --local i18n.logoutputencoding utf-8
git config --local core.quotepath false

echo Git encoding configured.
echo.
echo To fix commit messages, run:
echo   git rebase -i HEAD~4
echo.
echo Then change 'pick' to 'reword' for commits you want to fix.
echo After rebase, force push with: git push -f origin master
echo.
pause

