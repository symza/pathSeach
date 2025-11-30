import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from collections import defaultdict
import heapq
import matplotlib
from matplotlib.colors import to_rgb

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def interpolate_color(color1, color2, alpha):
    """在两个颜色之间插值
    alpha: 0.0 表示 color1, 1.0 表示 color2
    """
    rgb1 = np.array(to_rgb(color1))
    rgb2 = np.array(to_rgb(color2))
    rgb = rgb1 * (1 - alpha) + rgb2 * alpha
    return tuple(rgb)

def euclidean_distance(pos1, pos2):
    """计算两点之间的欧几里得距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def manhattan_distance(pos1, pos2):
    """计算两点之间的曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class AStarVisualization:
    def __init__(self, start_node=0, goal_node=6, heuristic_type='euclidean'):
        """
        初始化A*算法可视化
        
        参数:
            start_node: 起始节点
            goal_node: 目标节点
            heuristic_type: 启发式函数类型 ('euclidean' 或 'manhattan')
        """
        # 创建有向图
        self.G = nx.DiGraph()
        
        # 添加边（源节点，目标节点，权重）
        edges = [
            (0, 2, 2),
            (0, 1, 5),
            (2, 5, 8),
            (2, 3, 6),
            (1, 3, 1),
            (1, 4, 6),
            (3, 5, 2),
            (3, 4, 1),
            (5, 6, 3),
            (4, 6, 1)
        ]
        
        self.G.add_weighted_edges_from(edges)
        
        # 设置节点位置（使用spring布局，但固定位置）
        self.pos = {
            0: (0, 0),
            1: (2, 1),
            2: (2, -1),
            3: (4, 0),
            4: (4, 1),
            5: (6, -1),
            6: (6, 0)
        }
        
        self.start_node = start_node
        self.goal_node = goal_node
        self.heuristic_type = heuristic_type
        
        # 初始化距离字典
        self.g_scores = {node: float('inf') for node in self.G.nodes()}
        self.g_scores[start_node] = 0  # 源点距离为0
        
        # 计算启发式值
        self.h_scores = {}
        goal_pos = self.pos[goal_node]
        for node in self.G.nodes():
            node_pos = self.pos[node]
            if heuristic_type == 'euclidean':
                self.h_scores[node] = euclidean_distance(node_pos, goal_pos)
            else:
                self.h_scores[node] = manhattan_distance(node_pos, goal_pos)
        
        # 记录每一步的状态
        self.steps = []
        self.current_step = 0
        
        # 执行A*算法并记录每一步
        self.run_astar()
        
        # 扩展步骤以增加帧数到500
        self.expand_steps_to_500()
        
        # 创建图形（增大尺寸以提高清晰度）
        self.fig, self.ax = plt.subplots(figsize=(16, 10), dpi=200)
        self.ax.set_facecolor('#f0f0f0')
        
    def run_astar(self):
        """执行A*算法并记录每一步的状态"""
        # 初始化
        visited = set()
        previous = {}  # 记录每个节点的最短路径来源
        # 优先级队列：(f_score, g_score, node)
        # f_score = g_score + h_score
        pq = [(self.h_scores[self.start_node], 0, self.start_node)]
        processed_nodes = set()
        old_previous = {}  # 记录更新前的路径，用于标记变灰的边
        
        # 记录初始状态
        self.steps.append({
            'g_scores': self.g_scores.copy(),
            'h_scores': self.h_scores.copy(),
            'current_node': None,
            'visited': visited.copy(),
            'processed': processed_nodes.copy(),
            'previous': previous.copy(),
            'message': f'初始化：起点{self.start_node}的g(n)=0，h(n)={self.h_scores[self.start_node]:.1f}，f(n)={self.h_scores[self.start_node]:.1f}'
        })
        
        while pq:
            f_score, current_g, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            processed_nodes.add(current_node)
            
            # 记录选择当前节点
            self.steps.append({
                'g_scores': self.g_scores.copy(),
                'h_scores': self.h_scores.copy(),
                'current_node': current_node,
                'visited': visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': previous.copy(),
                'message': f'选择节点 {current_node}（g(n)={current_g:.1f}，h(n)={self.h_scores[current_node]:.1f}，f(n)={f_score:.1f}）'
            })
            
            # 如果到达目标节点，结束
            if current_node == self.goal_node:
                self.steps.append({
                    'g_scores': self.g_scores.copy(),
                    'h_scores': self.h_scores.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'processed': processed_nodes.copy(),
                    'previous': previous.copy(),
                    'message': f'✓ 找到目标节点 {self.goal_node}！最短路径长度为 {current_g:.1f}'
                })
                break
            
            # 更新邻居节点
            for neighbor in self.G.neighbors(current_node):
                if neighbor in visited:
                    continue
                
                edge_weight = self.G[current_node][neighbor]['weight']
                tentative_g = current_g + edge_weight
                
                # 记录检查边的过程（添加多个帧来显示线条逐渐延伸）
                self.steps.append({
                    'g_scores': self.g_scores.copy(),
                    'h_scores': self.h_scores.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'processed': processed_nodes.copy(),
                    'checking_edge': (current_node, neighbor),
                    'edge_progress': 0.0,
                    'previous': previous.copy(),
                    'message': f'开始检查边 {current_node}→{neighbor}（权重：{edge_weight}）'
                })
                
                # 添加线条逐渐延伸的中间帧
                current_previous = previous.copy()
                for progress in [i/15.0 for i in range(1, 15)]:
                    self.steps.append({
                        'g_scores': self.g_scores.copy(),
                        'h_scores': self.h_scores.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'checking_edge': (current_node, neighbor),
                        'edge_progress': progress,
                        'previous': current_previous.copy(),
                        'message': f'检查边 {current_node}→{neighbor}（权重：{edge_weight}）'
                    })
                
                # 添加检查完成的帧
                self.steps.append({
                    'g_scores': self.g_scores.copy(),
                    'h_scores': self.h_scores.copy(),
                    'current_node': current_node,
                    'visited': visited.copy(),
                    'processed': processed_nodes.copy(),
                    'checking_edge': (current_node, neighbor),
                    'edge_progress': 1.0,
                    'previous': previous.copy(),
                    'message': f'检查边 {current_node}→{neighbor}（权重：{edge_weight}）'
                })
                
                if tentative_g < self.g_scores[neighbor]:
                    old_g = self.g_scores[neighbor]
                    # 记录旧路径（如果存在）
                    old_prev = previous.get(neighbor)
                    if old_prev is not None:
                        old_previous[neighbor] = old_prev
                    
                    # 更新g值
                    self.g_scores[neighbor] = tentative_g
                    previous[neighbor] = current_node
                    
                    # 计算f值并加入优先队列
                    f_score_new = tentative_g + self.h_scores[neighbor]
                    heapq.heappush(pq, (f_score_new, tentative_g, neighbor))
                    
                    # 记录更新距离
                    step_data = {
                        'g_scores': self.g_scores.copy(),
                        'h_scores': self.h_scores.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'updated_node': neighbor,
                        'old_g_score': old_g,
                        'new_g_score': tentative_g,
                        'previous': previous.copy(),
                        'old_previous': old_previous.copy() if old_prev is not None else {},
                        'message': f'更新节点 {neighbor}：g(n)={old_g if old_g == float("inf") else old_g:.1f}→{tentative_g:.1f}，h(n)={self.h_scores[neighbor]:.1f}，f(n)={f_score_new:.1f}'
                    }
                    self.steps.append(step_data)
                else:
                    # 记录不更新
                    self.steps.append({
                        'g_scores': self.g_scores.copy(),
                        'h_scores': self.h_scores.copy(),
                        'current_node': current_node,
                        'visited': visited.copy(),
                        'processed': processed_nodes.copy(),
                        'previous': previous.copy(),
                        'message': f'节点 {neighbor} 距离未改善（当前g(n)={self.g_scores[neighbor]:.1f}，新g(n)={tentative_g:.1f}）'
                    })
        
        # 记录完成状态
        if self.goal_node in visited:
            # 重建路径
            path = []
            current = self.goal_node
            while current is not None:
                path.append(current)
                current = previous.get(current)
            path.reverse()
            
            self.steps.append({
                'g_scores': self.g_scores.copy(),
                'h_scores': self.h_scores.copy(),
                'current_node': None,
                'visited': visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': previous.copy(),
                'final_path': path,
                'message': f'算法完成！最短路径：{" → ".join(map(str, path))}，总长度：{self.g_scores[self.goal_node]:.1f}'
            })
        else:
            self.steps.append({
                'g_scores': self.g_scores.copy(),
                'h_scores': self.h_scores.copy(),
                'current_node': None,
                'visited': visited.copy(),
                'processed': processed_nodes.copy(),
                'previous': previous.copy(),
                'message': '算法完成！未找到到目标节点的路径'
            })
    
    def expand_steps_to_500(self):
        """扩展步骤以增加帧数到500，添加过渡帧和重复帧"""
        expanded_steps = []
        target_frames = 500
        base_steps = len(self.steps)
        
        if base_steps == 0:
            return
        
        # 为重要步骤分配更多帧数
        important_indices = {0, base_steps - 1}  # 初始化和完成
        
        # 识别重要步骤
        for i, step in enumerate(self.steps):
            if step.get('current_node') is not None and 'checking_edge' not in step:
                important_indices.add(i)
            if 'updated_node' in step:
                important_indices.add(i)
        
        # 计算基础帧数分配
        base_frames_per_step = max(3, (target_frames - len(important_indices) * 10) // base_steps)
        
        for i, step in enumerate(self.steps):
            # 重要步骤分配更多帧
            if i in important_indices:
                frames_for_this_step = base_frames_per_step + 5
            else:
                frames_for_this_step = base_frames_per_step
            
            # 添加重复帧
            for repeat in range(frames_for_this_step):
                if repeat == 0:
                    expanded_steps.append(step.copy())
                else:
                    expanded_steps.append(step.copy())
            
            # 在步骤之间添加过渡帧
            if i < len(self.steps) - 1:
                next_step = self.steps[i + 1]
                num_transitions = 6
                for transition in range(num_transitions):
                    alpha = (transition + 1) / (num_transitions + 1.0)
                    transition_step = self.create_transition_step(step, next_step, alpha)
                    expanded_steps.append(transition_step)
        
        # 调整到精确的500帧
        current_frames = len(expanded_steps)
        if current_frames < target_frames:
            while len(expanded_steps) < target_frames:
                expanded_steps.append(self.steps[-1].copy())
        elif current_frames > target_frames:
            indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
            expanded_steps = [expanded_steps[i] for i in indices]
        
        self.steps = expanded_steps[:target_frames]
    
    def create_transition_step(self, step1, step2, alpha):
        """创建两个步骤之间的过渡步骤"""
        transition = step1.copy()
        
        transition['transition_alpha'] = alpha
        transition['prev_step'] = step1
        transition['next_step'] = step2
        
        # 处理边的进度过渡
        if 'checking_edge' in step2 and 'edge_progress' in step2:
            transition['checking_edge'] = step2['checking_edge']
            if 'checking_edge' in step1 and step1['checking_edge'] == step2['checking_edge']:
                progress1 = step1.get('edge_progress', 0.0)
                progress2 = step2.get('edge_progress', 1.0)
                transition['edge_progress'] = progress1 * (1 - alpha) + progress2 * alpha
            else:
                transition['edge_progress'] = step2.get('edge_progress', 0.0) * alpha
        elif 'checking_edge' in step1 and 'edge_progress' in step1:
            transition['checking_edge'] = step1['checking_edge']
            transition['edge_progress'] = step1.get('edge_progress', 1.0)
        
        # 如果步骤2有新的更新，在过渡中显示
        if 'updated_node' in step2 and 'updated_node' not in step1:
            transition['updated_node'] = step2.get('updated_node')
            if 'g_scores' in step2:
                transition['g_scores'] = {}
                for node in step2['g_scores']:
                    g1 = step1['g_scores'].get(node, float('inf'))
                    g2 = step2['g_scores'].get(node, float('inf'))
                    if g1 == float('inf') and g2 == float('inf'):
                        transition['g_scores'][node] = float('inf')
                    elif g1 == float('inf'):
                        transition['g_scores'][node] = g2
                    elif g2 == float('inf'):
                        transition['g_scores'][node] = g1
                    else:
                        transition['g_scores'][node] = g1 * (1 - alpha) + g2 * alpha
            transition['message'] = step2.get('message', '')
        
        # 更新当前节点
        if 'current_node' in step2:
            transition['current_node'] = step2.get('current_node')
        
        # 更新已处理的节点
        if 'processed' in step2:
            transition['processed'] = step2['processed'].copy()
        
        return transition
    
    def draw_graph(self, step_data):
        """绘制当前步骤的图"""
        self.ax.clear()
        self.ax.set_facecolor('#f0f0f0')
        self.ax.set_xlim(-1, 7)
        self.ax.set_ylim(-2.5, 2)
        self.ax.axis('off')
        
        g_scores = step_data['g_scores']
        h_scores = step_data['h_scores']
        current_node = step_data.get('current_node')
        processed = step_data.get('processed', set())
        checking_edge = step_data.get('checking_edge')
        edge_progress = step_data.get('edge_progress', 1.0)
        updated_node = step_data.get('updated_node')
        transition_alpha = step_data.get('transition_alpha', 0.0)
        prev_step = step_data.get('prev_step')
        next_step = step_data.get('next_step')
        previous = step_data.get('previous', {})
        old_previous = step_data.get('old_previous', {})
        final_path = step_data.get('final_path', [])
        
        # 绘制边
        for edge in self.G.edges():
            u, v = edge
            weight = self.G[u][v]['weight']
            
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
                start_x = x1 + 0.15 * dx_norm
                start_y = y1 + 0.15 * dy_norm
                end_x = x2 - 0.15 * dx_norm
                end_y = y2 - 0.15 * dy_norm
                
                is_checking = checking_edge and checking_edge == (u, v)
                
                if is_checking:
                    partial_length_x = (end_x - start_x) * edge_progress
                    partial_length_y = (end_y - start_y) * edge_progress
                    partial_end_x = start_x + partial_length_x
                    partial_end_y = start_y + partial_length_y
                    
                    self.ax.plot([start_x, end_x], [start_y, end_y],
                               color='gray', linewidth=3, alpha=0.3, linestyle='--', zorder=1)
                    
                    if edge_progress < 1.0:
                        self.ax.plot([start_x, partial_end_x], [start_y, partial_end_y],
                                   color='red', linewidth=6, alpha=0.9, zorder=3, solid_capstyle='round')
                        self.ax.plot(partial_end_x, partial_end_y, 'ro', 
                                   markersize=12, zorder=4, alpha=0.9)
                    else:
                        self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                       arrowprops=dict(arrowstyle='->', lw=6, 
                                                     color='red', alpha=0.9, zorder=3))
                else:
                    # 判断边是否在最终路径中
                    is_in_final_path = False
                    if final_path:
                        for i in range(len(final_path) - 1):
                            if final_path[i] == u and final_path[i+1] == v:
                                is_in_final_path = True
                                break
                    
                    is_in_shortest_path = previous.get(v) == u
                    is_in_old_path = old_previous.get(v) == u
                    
                    if is_in_final_path:
                        # 最终路径，显示为紫色加粗
                        edge_color = 'purple'
                        edge_width = 5
                        linestyle = '-'
                        alpha = 1.0
                    elif is_in_shortest_path:
                        edge_color = 'green'
                        edge_width = 4
                        linestyle = '-'
                        alpha = 0.9
                    elif is_in_old_path:
                        edge_color = 'gray'
                        edge_width = 3
                        linestyle = '--'
                        alpha = 0.4
                    else:
                        edge_color = 'gray'
                        edge_width = 3
                        linestyle = '-'
                        alpha = 0.5
                    
                    self.ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                   arrowprops=dict(arrowstyle='->', lw=edge_width, 
                                                 color=edge_color, alpha=alpha,
                                                 linestyle=linestyle))
                
                # 添加权重标签
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                edges_below = [(2, 5, 8), (1, 3, 1), (3, 5, 2)]
                is_below = (u, v, weight) in edges_below
                
                if is_below:
                    offset_x = -dy_norm * 0.25
                    offset_y = -abs(dx_norm) * 0.2 - 0.1
                else:
                    offset_x = -dy_norm * 0.2
                    offset_y = dx_norm * 0.2
                
                self.ax.text(mid_x + offset_x, mid_y + offset_y, str(weight),
                           fontsize=20, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   edgecolor='gray', alpha=0.9, linewidth=3))
        
        # 绘制节点
        for node in self.G.nodes():
            x, y = self.pos[node]
            
            # 确定节点颜色（支持过渡）
            if transition_alpha > 0 and prev_step and next_step:
                prev_current = prev_step.get('current_node')
                prev_processed = prev_step.get('processed', set())
                prev_updated = prev_step.get('updated_node')
                
                if node == self.start_node:
                    prev_color = 'orange'
                elif node == self.goal_node:
                    prev_color = 'red'
                elif node == prev_current:
                    prev_color = 'yellow'
                elif node in prev_processed:
                    prev_color = 'lightgreen'
                elif prev_updated == node:
                    prev_color = 'lightblue'
                else:
                    prev_color = 'white'
                
                next_current = next_step.get('current_node') if next_step else current_node
                next_processed = next_step.get('processed', set()) if next_step else processed
                next_updated = next_step.get('updated_node') if next_step else updated_node
                
                if node == self.start_node:
                    next_color = 'orange'
                elif node == self.goal_node:
                    next_color = 'red'
                elif node == next_current:
                    next_color = 'yellow'
                elif node in next_processed:
                    next_color = 'lightgreen'
                elif next_updated == node:
                    next_color = 'lightblue'
                else:
                    next_color = 'white'
                
                node_color = interpolate_color(prev_color, next_color, transition_alpha)
            else:
                # 正常颜色
                if node == self.start_node:
                    node_color = 'orange'  # 起点
                elif node == self.goal_node:
                    node_color = 'red'  # 终点
                elif node == current_node:
                    node_color = 'yellow'  # 当前处理的节点
                elif node in processed:
                    node_color = 'lightgreen'  # 已处理的节点
                elif updated_node == node:
                    node_color = 'lightblue'  # 刚更新的节点
                else:
                    node_color = 'white'  # 未处理的节点
            
            # 绘制节点圆圈
            if isinstance(node_color, tuple):
                node_color_final = node_color
            else:
                node_color_final = node_color
            
            circle = plt.Circle((x, y), 0.3, color=node_color_final, 
                              ec='black', lw=4, zorder=3)
            self.ax.add_patch(circle)
            
            # 添加节点编号
            if node == self.start_node or node == self.goal_node:
                text_color = 'white'
            elif isinstance(node_color, tuple):
                brightness = sum(node_color) / 3.0
                text_color = 'black' if brightness > 0.5 else 'white'
            else:
                if node_color in ['white', 'yellow', 'lightgreen', 'lightblue']:
                    text_color = 'black'
                else:
                    text_color = 'white'
            
            self.ax.text(x, y, str(node), fontsize=26, fontweight='bold',
                        ha='center', va='center', zorder=4, color=text_color)
            
            # 添加g(n)和f(n)标签
            g_score = g_scores[node]
            h_score = h_scores[node]
            f_score = g_score + h_score if g_score != float('inf') else float('inf')
            
            # 格式化显示
            if g_score == float('inf'):
                g_text = 'Inf'
                f_text = 'Inf'
            else:
                g_text = f'{g_score:.1f}'
                f_text = f'{f_score:.1f}'
            
            # 根据节点位置调整标签位置
            if node == 4 or node == 6:
                label_x = x + 0.4
                label_y = y - 0.4
                ha_align = 'left'
                va_align = 'bottom'
            else:
                label_x = x
                label_y = y - 0.6
                ha_align = 'center'
                va_align = 'center'
            
            # 显示g(n)和f(n)
            label_text = f'g={g_text}\nf={f_text}'
            if node == self.start_node or node == self.goal_node or node == current_node or updated_node == node:
                label_color = 'blue'
            else:
                label_color = 'red'
            
            self.ax.text(label_x, label_y, label_text, fontsize=18, fontweight='bold',
                        ha=ha_align, va=va_align, color=label_color,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor=label_color, alpha=0.8, linewidth=3))
            
            # 如果是起点或终点，添加标签
            if node == self.start_node:
                self.ax.text(x - 0.5, y, 'start', fontsize=18, fontweight='bold',
                           ha='right', va='center', color='red', style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='red', alpha=0.7, linewidth=2.5))
            elif node == self.goal_node:
                self.ax.text(x + 0.5, y, 'goal', fontsize=18, fontweight='bold',
                           ha='left', va='center', color='red', style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='red', alpha=0.7, linewidth=2.5))
        
        # 添加步骤信息
        message = step_data.get('message', '')
        self.ax.text(3, -2.2, message, fontsize=20, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=1.0', facecolor='yellow', 
                            edgecolor='black', alpha=0.95, linewidth=4),
                    fontweight='bold')
        
        # 添加图例
        legend_text = "图例：\n橙色=起点 | 红色=终点 | 黄色=当前节点 | 绿色=已处理 | 蓝色=刚更新\n紫色边=最终路径 | 绿色边=最短路径 | 红色边=正在检查"
        self.ax.text(0.2, 1.9, legend_text, fontsize=16, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                            edgecolor='gray', alpha=0.9, linewidth=3),
                    fontweight='bold')
    
    def animate(self, frame):
        """动画函数"""
        if frame < len(self.steps):
            self.draw_graph(self.steps[frame])
        return []
    
    def create_animation(self, output_file='astar_animation.gif', interval=100, dpi=200):
        """创建动画并保存为GIF
        
        参数:
            output_file: 输出文件名
            interval: 每帧间隔时间（毫秒）
            dpi: 图像分辨率
        """
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 12
        
        total_frames = len(self.steps)
        print(f"正在生成动画，共 {total_frames} 帧...")
        print(f"这可能需要一些时间，请稍候...")
        print(f"提示：生成高分辨率动画需要较长时间，请耐心等待...")
        
        anim = animation.FuncAnimation(self.fig, self.animate, 
                                      frames=total_frames,
                                      interval=interval, 
                                      repeat=True, 
                                      blit=False)
        
        fps = 1000 / interval if interval > 0 else 10
        anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
        
        print(f"\n✓ 动画已成功保存为: {output_file}")
        print(f"  - 总帧数: {len(self.steps)}")
        print(f"  - 分辨率: {dpi} DPI")
        print(f"  - 每帧间隔: {interval}ms")
        
        return anim

if __name__ == '__main__':
    print("=" * 60)
    print("A*算法动态演示生成器")
    print("=" * 60)
    
    # 创建可视化对象
    print("\n正在初始化图结构...")
    print("起点: 0, 终点: 6")
    print("启发式函数: 欧几里得距离")
    viz = AStarVisualization(start_node=0, goal_node=6, heuristic_type='euclidean')
    
    # 生成动画
    print("\n开始生成动画...")
    anim = viz.create_animation('astar_demo.gif', interval=100, dpi=200)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


