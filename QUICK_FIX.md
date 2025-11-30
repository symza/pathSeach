# 快速修复提交信息乱码

## 问题
GitHub 网页上提交信息显示乱码，因为提交时使用了错误的编码。

## 快速修复方法

### 方法一：使用交互式 Rebase（推荐）

1. **打开命令行**，切换到项目目录：
   ```bash
   cd D:\Algorithm_analysis
   ```

2. **设置控制台编码**：
   ```bash
   chcp 65001
   ```

3. **开始交互式 rebase**：
   ```bash
   git rebase -i HEAD~4
   ```

4. **在编辑器中**，将需要修改的提交前的 `pick` 改为 `reword`（或简写 `r`）：
   ```
   pick bf45cab 初始提交（这个保持 pick）
   reword 7bc16f2 添加 README（改为 reword）
   reword 1b76e68 配置编码（改为 reword）
   pick 3475f29 Add Git encoding（这个保持 pick）
   ```

5. **保存并关闭编辑器**（在 Vim 中：按 `Esc`，输入 `:wq`，按回车）

6. **逐个修改提交信息**：
   - 对于 `7bc16f2`：输入 `添加 README.md 项目说明文档`
   - 对于 `1b76e68`：输入 `配置 Git 编码和文件属性，修复中文显示问题`
   - 每次修改后保存关闭

7. **完成 rebase 后，强制推送**：
   ```bash
   git push -f origin master
   ```

### 方法二：逐个修改（如果方法一失败）

#### 修改最近的提交
```bash
git commit --amend -m "配置 Git 编码和文件属性，修复中文显示问题"
git push -f origin master
```

#### 修改更早的提交
需要用到 rebase，参考方法一。

## 正确的提交信息

- **初始提交**：`初始提交：路径搜索算法演示程序 - 包含Dijkstra、A*、RRT三种算法实现`
- **添加 README**：`添加 README.md 项目说明文档`
- **配置编码**：`配置 Git 编码和文件属性，修复中文显示问题`

## 注意事项

⚠️ **强制推送警告**：
- `git push -f` 会覆盖远程仓库的历史
- 如果其他人也在使用这个仓库，请先通知他们
- 建议在修复前先备份

## 如果不想修改历史

如果不想修改已推送的提交历史，可以：
1. 保持现状（不影响功能）
2. 未来的提交使用正确的编码即可
3. 在 README 中添加说明

