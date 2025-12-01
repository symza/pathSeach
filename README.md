# 路径搜索算法演示程序

一个基于 Python 和 Tkinter 的图形化路径搜索算法演示程序集合，支持多种算法的可视化演示。

## 功能特性

### 🎯 支持的算法

1. **Dijkstra 算法**
   - 经典的最短路径算法
   - 保证找到最短路径
   - 支持网格迷宫可视化

2. **A* 算法**
   - 启发式搜索算法
   - 可调整启发式权重
   - 平衡搜索速度和路径质量

3. **RRT 算法** (Rapidly-exploring Random Tree)
   - 快速随机扩展树算法
   - 适合复杂环境下的路径规划
   - 可调整步长、目标偏向度、最大迭代次数

4. **综合演示程序** (Pathfinding Maze App)
   - 集成 Dijkstra、A*、RRT 三种算法
   - 统一的图形界面
   - 支持算法切换和参数调整

### 🎨 可视化功能

- 实时显示算法搜索过程
- 可调整迷宫大小、复杂度、密度
- 动画速度控制
- 支持暂停/继续
- 显示搜索步骤和路径信息

### 🎬 动画生成

- 自动生成算法演示 GIF 动画
- 播放时间自动优化（限制在 10 秒内）
- 支持导出为 GIF 文件
- 自动采样优化帧数

## 项目结构

```
Algorithm_analysis/
├── a_star/                    # A* 算法演示
│   ├── src/
│   │   ├── astar_interactive.py      # 交互式演示程序
│   │   ├── astar_animation.py        # 动画生成程序
│   │   └── build_astar_interactive.py # 打包脚本
│   └── images/                # 生成的动画文件
│
├── dijkstra/                  # Dijkstra 算法演示
│   ├── src/
│   │   ├── dijkstra_interactive.py   # 交互式演示程序
│   │   ├── dijkstra_animation.py     # 动画生成程序
│   │   ├── maze_dijkstra.py          # 迷宫生成工具
│   │   ├── maze_dijkstra_app.py      # 迷宫应用
│   │   └── build_dijkstra_interactive.py # 打包脚本
│   └── images/                # 生成的动画文件
│
├── rrt/                        # RRT 算法演示
│   ├── src/
│   │   ├── rrt_interactive.py        # 交互式演示程序
│   │   ├── rrt_animation.py          # 动画生成程序
│   │   └── build_rrt_interactive.py  # 打包脚本
│   └── images/                # 生成的动画文件
│
├── pathfinding/                # 综合路径搜索演示
│   ├── src/
│   │   ├── pathfinding_maze_app.py   # 主程序（集成三种算法）
│   │   └── build_pathfinding_maze_app.py # 打包脚本
│   └── images/                # 生成的动画文件
│
├── .gitignore                 # Git 忽略规则
├── .gitattributes            # Git 属性配置
└── README.md                  # 项目说明文档
```

## 快速开始

### 环境要求

- Python 3.8+
- Windows 7+（打包后的 exe 文件）
- 所需 Python 包见下方依赖列表

### 安装依赖

主要依赖包：
- `numpy` - 数值计算
- `matplotlib` - 图形绘制
- `Pillow` (PIL) - 图像处理
- `PyInstaller` - 打包工具（可选，用于生成 exe）

安装命令：
```bash
pip install numpy matplotlib pillow pyinstaller
```

### 运行程序

#### 1. 运行综合演示程序（推荐）

```bash
cd pathfinding/src
python pathfinding_maze_app.py
```

#### 2. 运行单个算法演示

**A* 算法：**
```bash
cd a_star/src
python astar_interactive.py
```

**Dijkstra 算法：**
```bash
cd dijkstra/src
python dijkstra_interactive.py
```

**RRT 算法：**
```bash
cd rrt/src
python rrt_interactive.py
```

### 生成动画

每个算法都有对应的动画生成脚本：

```bash
# A* 动画
cd a_star/src
python astar_animation.py

# Dijkstra 动画
cd dijkstra/src
python dijkstra_animation.py

# RRT 动画
cd rrt/src
python rrt_animation.py
```

生成的 GIF 文件会保存在各自的 `images/` 文件夹中。

### 打包为可执行文件

每个算法都有独立的打包脚本：

```bash
# 打包 A* 演示程序
cd a_star/src
python build_astar_interactive.py

# 打包 Dijkstra 演示程序
cd dijkstra/src
python build_dijkstra_interactive.py

# 打包 RRT 演示程序
cd rrt/src
python build_rrt_interactive.py

# 打包综合演示程序
cd pathfinding/src
python build_pathfinding_maze_app.py
```

打包完成后，exe 文件会生成在各自的文件夹下（如 `a_star/AStarInteractive.exe`）。

**注意：** 打包脚本会询问是否创建调试版本：
- **调试版本**：显示控制台窗口，便于查看错误信息（推荐首次打包）
- **正式版本**：不显示控制台窗口，界面更简洁

## 使用说明

### 综合演示程序操作

1. **生成迷宫**
   - 点击"生成新迷宫"按钮
   - 调整大小、复杂度、密度参数

2. **选择算法**
   - 选择要使用的路径搜索算法（Dijkstra、A*、RRT）
   - 根据选择的算法调整相应参数

3. **算法参数**
   - **A* 算法**：调整启发式权重（越大越倾向于目标）
   - **RRT 算法**：调整步长因子、目标偏向度、最大迭代次数

4. **开始搜索**
   - 点击"开始搜索"查看算法运行过程
   - 使用"暂停/继续"控制动画
   - 调整动画速度滑块

5. **生成动图**
   - 点击"生成动图"导出 GIF 动画
   - 文件会自动保存到 `images/` 文件夹

### 交互式演示程序操作

- **鼠标操作**：点击设置起点和终点
- **键盘操作**：
  - `空格键`：单步执行
  - `回车键`：自动运行
  - `R`：重置
  - `S`：保存当前状态
  - `Q`：退出

## 算法说明

### Dijkstra 算法

经典的最短路径算法，使用广度优先搜索的思想，保证找到最短路径。

**特点：**
- 保证最优解
- 适用于网格地图
- 时间复杂度：O(V²) 或 O(E log V)

### A* 算法

启发式搜索算法，结合了 Dijkstra 的最优性和贪心搜索的高效性。

**特点：**
- 使用启发式函数引导搜索
- 可调整启发式权重平衡速度和最优性
- 当权重为 1 时，保证找到最短路径

**参数说明：**
- **启发式权重**：权重越大，算法越倾向于向目标移动（更快但可能不是最优）

### RRT 算法

快速随机扩展树算法，通过随机采样构建搜索树，适合高维空间和复杂环境。

**特点：**
- 概率完备性（给定足够时间能找到解）
- 适合连续空间路径规划
- 路径不唯一

**参数说明：**
- **步长因子**：越小步长越大（默认 6.0）
- **目标偏向度**：越高越倾向目标（0-1，默认 0.3）
- **最大迭代次数**：越多成功率越高（默认 5000）

## 技术实现

- **图形界面**：Tkinter + Matplotlib
- **算法实现**：Python 标准库（heapq、collections）
- **动画生成**：Matplotlib Animation + PIL
- **打包工具**：PyInstaller

## 注意事项

1. **exe 文件**：生成的 exe 文件较大（约 50-100MB），包含所有依赖，无需安装 Python 环境
2. **文件大小**：exe 文件不会提交到 Git，需要通过打包脚本生成
3. **平台支持**：打包脚本主要针对 Windows，其他平台需要修改 PyInstaller 参数
4. **性能**：大型迷宫（>60x60）可能运行较慢，建议使用较小的迷宫进行演示

## 许可证

本项目仅供学习和演示使用。

## 贡献

欢迎提交 Issue 和 Pull Request！
