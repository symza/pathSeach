# 修复 GitHub 提交信息乱码

## 问题
GitHub 网页上提交信息显示乱码，因为提交时使用了错误的编码。

## 快速修复步骤

### 步骤 1：打开命令行并设置编码

```bash
cd D:\Algorithm_analysis
chcp 65001
```

### 步骤 2：配置 Git 编码

```bash
git config --local i18n.commitencoding utf-8
git config --local i18n.logoutputencoding utf-8
git config --local core.quotepath false
```

### 步骤 3：开始交互式 Rebase

```bash
git rebase -i HEAD~4
```

### 步骤 4：在编辑器中修改

将需要修复的提交前的 `pick` 改为 `reword`（或简写 `r`）：

```
pick bf45cab 初始提交（保持 pick）
reword 7bc16f2 添加 README（改为 reword）
reword 1b76e68 配置编码（改为 reword）
pick 3475f29 Add Git encoding（保持 pick）
```

保存并关闭编辑器（Vim: 按 `Esc`，输入 `:wq`，回车）

### 步骤 5：逐个修改提交信息

对于每个标记为 `reword` 的提交，Git 会打开编辑器：

- **7bc16f2**: 输入 `添加 README.md 项目说明文档`
- **1b76e68**: 输入 `配置 Git 编码和文件属性，修复中文显示问题`
- **bf45cab**: 输入 `初始提交：路径搜索算法演示程序 - 包含Dijkstra、A*、RRT三种算法实现`

每次修改后保存关闭。

### 步骤 6：强制推送

```bash
git push -f origin master
```

⚠️ **警告**：强制推送会覆盖远程历史，请确认后再执行！

## 如果不想修改历史

如果不想修改已推送的提交历史：
1. 保持现状（不影响功能）
2. 未来提交使用正确编码即可
3. 在 README 中添加说明

## 验证修复

修复后，访问 https://github.com/symza/pathSeach 查看，提交信息应该正常显示中文。

