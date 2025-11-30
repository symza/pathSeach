# Git 中文编码配置说明

## 解决中文乱码问题

### 方法一：运行配置脚本（推荐）

双击运行 `fix_git_encoding.bat`，它会自动配置 Git 使用 UTF-8 编码。

### 方法二：手动配置

在命令行中运行以下命令：

```bash
# 配置全局 Git 编码
git config --global core.quotepath false
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
git config --global gui.encoding utf-8

# 配置本地仓库编码
git config --local core.quotepath false
git config --local i18n.commitencoding utf-8
git config --local i18n.logoutputencoding utf-8

# 设置控制台编码
chcp 65001
```

### 重要提示

1. **提交信息已保存为 UTF-8**：虽然在命令行可能显示乱码，但在 GitHub 网页上会正常显示
2. **文件内容正常**：所有源代码文件使用 UTF-8 编码，不会出现乱码
3. **重新打开命令行**：配置后需要重新打开命令行窗口才能生效

### 验证配置

运行以下命令验证：

```bash
git config --list | findstr encoding
```

## 常用 Git 命令

```bash
# 查看提交历史
git log --oneline

# 推送代码
git push origin master

# 拉取更新
git pull origin master

# 查看状态
git status
```

