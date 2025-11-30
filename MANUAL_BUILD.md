# 手动打包指南

如果批处理文件无法正常工作，请按照以下步骤手动打包：

## 步骤 1：打开命令行

- 按 `Win + R`
- 输入 `cmd` 或 `powershell`
- 按回车

## 步骤 2：切换到项目目录

```bash
cd D:\Algorithm_analysis
```

## 步骤 3：运行打包命令

### 简化版（快速测试）：

```bash
python -m PyInstaller --windowed --onefile --name PathfindingMazeDemo pathfinding_maze_app.py
```

### 完整版（推荐，包含所有依赖）：

```bash
python -m PyInstaller --name PathfindingMazeDemo --windowed --onefile --clean --noconfirm --hidden-import=tkinter --hidden-import=tkinter.ttk --hidden-import=tkinter.messagebox --hidden-import=tkinter.filedialog --hidden-import=matplotlib.backends.backend_tkagg --hidden-import=matplotlib.figure --hidden-import=PIL._tkinter_finder --hidden-import=numpy --hidden-import=matplotlib --hidden-import=PIL --hidden-import=io --collect-all=matplotlib --collect-all=PIL pathfinding_maze_app.py
```

## 步骤 4：等待打包完成

- 通常需要 2-5 分钟
- 会看到很多输出信息，这是正常的

## 步骤 5：找到可执行文件

打包完成后，在 `dist` 文件夹中找到：
- `PathfindingMazeDemo.exe`（简化版）
- 或 `路径搜索算法演示.exe`（完整版）

## 如果遇到错误

### 错误：找不到模块

添加隐藏导入：
```bash
--hidden-import=模块名
```

### 错误：缺少依赖

确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

## 复制完整命令（右键点击可全选）

完整打包命令（复制到命令行）：

```
python -m PyInstaller --name PathfindingMazeDemo --windowed --onefile --clean --noconfirm --hidden-import=tkinter --hidden-import=tkinter.ttk --hidden-import=tkinter.messagebox --hidden-import=tkinter.filedialog --hidden-import=matplotlib.backends.backend_tkagg --hidden-import=matplotlib.figure --hidden-import=PIL._tkinter_finder --hidden-import=numpy --hidden-import=matplotlib --hidden-import=PIL --hidden-import=io --collect-all=matplotlib --collect-all=PIL pathfinding_maze_app.py
```

