import math
import random
import heapq
import time
import sys
import os
# --- 全局常量和变量 ---
MINX = -1000
MAXX = 1000

D = 4  # 向量维度
M = 16  # 每个节点的最大连接数 (除第0层外)
MMAX0 = 2 * M  # 第0层的最大连接数
MMAX = M
N = 10000  # 数据点数量
EF_CONSTRUCTION = 10  # 构建索引时的搜索范围大小
K = 1  # K-NN搜索中的K值

# 数据和图结构
# 'node' 将在 init() 中加载或在 gendata() 后填充
node = [[0] * D for _ in range(N + 1)] # node[0] 不使用
edge = [[]]  # 邻接表, edge[0] 作为占位符
ite = [0]   # 索引到节点ID的映射, ite[0] 作为占位符

# HNSW 算法状态
L = 0  # 当前最高层数
EP = 1  # 入口点 (entry point) 的索引
nodecnt = 0  # 图中节点的总数 (跨所有层)
ML = 1 / math.log(M)

# --- 辅助函数 ---
def gendata(num=N, dim=D):
    """生成随机数据并写入 data.txt"""
    
    # --- 核心修改部分 ---
    # 1. 获取当前脚本文件所在的目录
    # __file__ 是一个特殊变量，代表当前脚本文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 使用 os.path.join 构建出目标文件的完整绝对路径
    #    os.path.join 会自动处理不同操作系统下的路径分隔符（/ 或 \）
    output_path = os.path.join(script_dir, "data.txt")
    # --- 结束修改 ---
    
    print("开始生成数据...", file=sys.stderr)
    print(f"将要写入文件到: {output_path}", file=sys.stderr) # 增加打印，方便确认路径
    
    # 使用构建好的绝对路径来打开文件
    with open(output_path, "w") as f:
        for i in range(num):
            line = ' '.join(str(random.randint(MINX, MAXX)) for _ in range(dim))
            f.write(line + "\n")
    print("完成写入 data.txt", file=sys.stderr)

def distance(q_idx, x_idx):
    """计算两个节点索引之间的欧氏距离"""
    nq = node[q_idx]
    nx = node[x_idx]
    res_sq = sum((nq[i] - nx[i]) ** 2 for i in range(D))
    return math.sqrt(res_sq)

def nxt(W):
    """
    将 W 中的图节点索引加一.
    在C++版本中，这是因为 ep 是指向下一层节点的指针/索引。
    在Python中，我们直接返回处理后的新列表。
    """
    return [i + 1 for i in W]

# --- HNSW 核心算法 ---

def search_layer(q_idx, ep, ef, lc):
    """
    在指定层 (lc) 上搜索最近的 ef 个邻居.
    q_idx: 查询点的原始节点ID
    ep: 入口点列表 (图中的索引)
    ef: 搜索范围大小
    lc: 当前层级
    """
    v = set(ep)
    # C: 候选小顶堆 (距离, 图索引)
    C = [(distance(q_idx, ite[x]), x) for x in ep]
    heapq.heapify(C)
    # W: 结果大顶堆 (-距离, 图索引), Python heapq 是小顶堆, 所以存入负值模拟
    W = [(-distance(q_idx, ite[x]), x) for x in ep]
    heapq.heapify(W)

    while C:
        dist_c, c_node_idx = heapq.heappop(C)

        if dist_c > -W[0][0]:
            break  # 所有最近的候选点都已被评估

        for e_node_idx in edge[c_node_idx]:
            if e_node_idx not in v:
                v.add(e_node_idx)
                dist_e = distance(q_idx, ite[e_node_idx])
                
                if len(W) < ef or dist_e < -W[0][0]:
                    heapq.heappush(C, (dist_e, e_node_idx))
                    heapq.heappush(W, (-dist_e, e_node_idx))
                    if len(W) > ef:
                        heapq.heappop(W)

    return [item[1] for item in sorted(W, key=lambda x: x[0], reverse=True)]


def select_neighbors_heuristic(q_idx, C, M_conn, lc, extend_candidates=False, keep_pruned_connections=True):
    """
    使用启发式规则从候选集 C 中为 q 选择 M_conn 个邻居.
    q_idx: 查询点的原始节点ID
    C: 候选点列表 (图中的索引)
    M_conn: 需要选择的邻居数量
    """
    R = [] # 结果小顶堆 (距离, 图索引)
    W = [] # 工作小顶堆 (距离, 图索引)

    for x in C:
        heapq.heappush(W, (distance(q_idx, ite[x]), x))

    # C++ 版本中的这部分逻辑在原始代码中是可选的，这里为了完全对应也实现
    if extend_candidates:
        st = set(C)
        for e in C:
            for e_adj in edge[e]:
                if e_adj not in st:
                    st.add(e_adj)
                    heapq.heappush(W, (distance(q_idx, ite[e_adj]), e_adj))
    
    Wd = [] # 被丢弃的候选者
    while W and len(R) < M_conn:
        dist_e, e_node_idx = heapq.heappop(W)
        
        # C++ 版本中 R 为空或新元素更近则插入的逻辑
        # Python heapq 没有 peek, 但可以通过 R[0] 实现
        if not R or dist_e < R[0][0]:
             heapq.heappush(R, (dist_e, e_node_idx))
        else:
             heapq.heappush(Wd, (dist_e, e_node_idx))

    if keep_pruned_connections:
        while Wd and len(R) < M_conn:
            heapq.heappush(R, heapq.heappop(Wd))
    
    return [item[1] for item in R]

def insert(q_idx, M_param, Mmax_param, efConstruction_param, mL_param):
    """将节点 q_idx 插入到 HNSW 图中"""
    global L, EP, nodecnt

    # 1. 初始化本次插入的入口点 (使用插入前的全局EP)
    #    C++: vector<int> ep; if (nodecnt != 0) ep.emplace_back(EP);
    current_ep = [EP] if nodecnt != 0 else []

    # 2. 为新节点随机选择层数
    l = math.floor(-math.log(random.uniform(0.0, 1.0)) * mL_param)

    # 3. 如果新节点的层级高于当前图，预先创建节点占位并更新全局 EP
    if l > L:
        # ---- BUG FIX STARTS HERE ----
        # 关键修正：必须先确定新入口点的索引，再创建节点。
        # EP 应该指向新节点在最高层(l层)的索引，即 nodecnt + 1。
        EP = nodecnt + 1
        # C++ ford(lc, l, L + 1) -> for(i=l; i>=L+1; --i)
        # Python range(l, L, -1) -> l, l-1, ..., L+1
        for _ in range(l, L, -1):
            nodecnt += 1
            ite.append(q_idx)
            edge.append([]) # 为新节点在新层添加空的邻居列表
        # ---- BUG FIX ENDS HERE ----

    # 4. 自顶向下搜索，找到每一层的最近邻居作为下一层的入口点
    #    这个搜索过程必须从本次插入操作开始前的入口点(current_ep)开始
    
    # 从图的最高层 L 向下搜索到 l+1 层
    for lc in range(L, l, -1):
        W = search_layer(q_idx, current_ep, 1, lc)
        if not W: # 如果在某高层找不到路径，则无法继续
            break
        assert len(W) == 1
        current_ep = nxt(W)

    # 5. 从 l 层开始，正式插入节点并建立双向连接
    for lc in range(min(L, l), -1, -1):
        m_max_lc = MMAX0 if lc == 0 else Mmax_param
        
        nodecnt += 1
        ite.append(q_idx)
        
        W = search_layer(q_idx, current_ep, efConstruction_param, lc)
        if not W: # 如果在某层找不到候选邻居，则从下一层开始
            edge.append([])
            continue

        neighbors = select_neighbors_heuristic(q_idx, W, M_param, lc)
        
        edge.append(neighbors)
        new_node_graph_idx = nodecnt
        
        # 为邻居节点添加入边 (建立双向连接)
        for e_node_idx in neighbors:
            edge[e_node_idx].append(new_node_graph_idx)
            
            # 如果邻居的连接数超了，进行修剪
            if len(edge[e_node_idx]) > m_max_lc:
                e_conn = edge[e_node_idx]
                e_new_conn = select_neighbors_heuristic(ite[e_node_idx], e_conn, m_max_lc, lc)
                edge[e_node_idx] = e_new_conn
        
        # 将本层的结果作为下一层的入口点
        current_ep = nxt(W)

    # 6. 如果新节点的层级更高，更新全局最高层 L
    if l > L:
        L = l

def k_nn_search(q_idx, k_param, ef_param):
    """在 HNSW 图中搜索 q_idx 的 K 个最近邻"""
    W = []
    ep = [EP]
    
    # 从顶层下探到第一层
    for lc in range(L, 0, -1):
        W = search_layer(q_idx, ep, 1, lc)
        assert len(W) == 1
        ep = nxt(W)

    # 在第0层进行更广泛的搜索
    W = search_layer(q_idx, ep, ef_param, 0)
    
    # 从结果中提取K个最近的 (使用大顶堆)
    result_heap = [] # (-距离, 原始节点ID)
    for x_node_idx in W:
        dist = distance(q_idx, ite[x_node_idx])
        heapq.heappush(result_heap, (-dist, ite[x_node_idx]))
        if len(result_heap) > k_param:
            heapq.heappop(result_heap)
            
    return [item[1] for item in sorted(result_heap, key=lambda x: x[0], reverse=True)]

# --- 初始化和主程序 ---
def init():
    """从文件加载数据"""
    
    # --- 核心修改部分 ---
    # 1. 获取当前脚本文件所在的目录的绝对路径
    #    __file__ 是一个特殊变量，代表当前脚本文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 使用 os.path.join 构建出 data.txt 的完整绝对路径
    #    这确保了无论从哪里运行脚本，路径总是正确的
    data_path = os.path.join(script_dir, "data.txt")
    # --- 结束修改 ---

    print("开始读取数据...", file=sys.stderr)
    # 增加一行打印，方便我们确认程序正在从正确的路径读取文件
    print(f"尝试从以下路径读取文件: {data_path}", file=sys.stderr)

    try:
        # 使用构建好的绝对路径来打开文件
        with open(data_path, "r") as f:
            for i in range(1, N + 1):
                line = f.readline()
                if not line: break
                # 假设 node 是一个已定义的字典或列表
                node[i] = [int(x) for x in line.strip().split()]
                
    except FileNotFoundError:
        # 更新错误信息，显示我们尝试查找的完整路径
        print(f"错误: 在路径 {data_path} 未找到 data.txt。", file=sys.stderr)
        print("请先运行 gendata() 生成数据文件。", file=sys.stderr)
        exit(1)
        
    print("完成读取 data.txt", file=sys.stderr)
    
    # 打印前3个节点以供验证 (这部分代码保持不变)
    # 确保在调用此函数前 node 已经被初始化为一个字典
    if len(node) >= 3:
        for i in range(1, 4):
            print(f"节点 {i}: {node[i]}", file=sys.stderr)


def bruteforce(q_idx):
    """暴力搜索 q_idx 的 K 个最近邻以供验证"""
    start_time = time.time()
    
    # 使用大顶堆来找到 K 个最近邻
    W = [] # (-距离, 原始节点ID)
    for i in range(1, N + 1):
        dist = distance(q_idx, i)
        heapq.heappush(W, (-dist, i))
        if len(W) > K:
            heapq.heappop(W)

    ret = sorted([item[1] for item in W], reverse=True)
    
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000

    print("\n暴力搜索结果:", file=sys.stderr)
    print(' '.join(map(str, ret)), file=sys.stderr)
    for r_idx in ret:
        dist = distance(r_idx, q_idx)
        print(f"{node[r_idx]} || {dist:.4f}", file=sys.stderr)
    print(f"查询耗时: {duration_ms:.4f} ms", file=sys.stderr)
    return ret


def main():
    """主执行函数"""
    # gendata()
    init()
    
    print("\n*********************", file=sys.stderr)
    print(f"N (数据点数): {N}", file=sys.stderr)
    print(f"D (维度): {D}", file=sys.stderr)
    print(f"M (最大连接数): {M}", file=sys.stderr)
    print(f"mL (层级因子): {ML:.4f}", file=sys.stderr)
    print(f"K (近邻数): {K}", file=sys.stderr)
    print(f"efConstruction: {EF_CONSTRUCTION}", file=sys.stderr)
    print("**********************\n", file=sys.stderr)

    start_insert = time.time()
    for i in range(1, N + 1):
        insert(i, M, MMAX, EF_CONSTRUCTION, ML)
        if i % (N // 100) == 0:
            print(f"已插入节点数 = {i} / {N}", file=sys.stderr)
            
    end_insert = time.time()
    duration_insert = end_insert - start_insert
    print("\nHNSW 图构建完成", file=sys.stderr)

    # --- 查询和评估 ---
    nq = 21 # 查询次数
    print(f"输入查询次数: {nq}", file=sys.stderr)

    # 生成查询点
    query_nodes_start_idx = N + 1
    for i in range(nq):
        query_node_idx = query_nodes_start_idx + i
        val = (i - (nq - 1) / 2) * ((MAXX - MINX) / (nq - 1))
        # 确保 node 列表足够大
        while len(node) <= query_node_idx:
            node.append([0] * D)
        node[query_node_idx] = [int(val)] * D
        
    print(f"\nnode.size: {len(node)}", file=sys.stderr)

    ave_recall = 0.0
    for i in range(nq):
        q_idx = query_nodes_start_idx + i
        
        print("\n--------------------------------", file=sys.stderr)
        q_vec_str = ", ".join(map(str, node[q_idx]))
        print(f"当前查询点 q 为 ({q_vec_str})", file=sys.stderr)
        
        print("\nq 的 K-ANN 搜索结果:", file=sys.stderr)
        start_query = time.time()
        knn_res = k_nn_search(q_idx, K, EF_CONSTRUCTION)
        end_query = time.time()
        duration_query_ms = (end_query - start_query) * 1000

        print(' '.join(map(str, knn_res)), file=sys.stderr)
        for res_idx in knn_res:
            dist = distance(res_idx, q_idx)
            print(f"{node[res_idx]} || {dist:.4f}", file=sys.stderr)
        print(f"查询耗时: {duration_query_ms:.4f} ms", file=sys.stderr)

        brt_res = bruteforce(q_idx)
        
        brt_set = set(brt_res)
        ccnt = sum(1 for res_idx in knn_res if res_idx in brt_set)
        recall = ccnt / K
        ave_recall += recall
        print(f"\n召回率: {recall:.4f}", file=sys.stderr)
        print("-------------------------------------\n", file=sys.stderr)
        
    print(f"平均召回率: {ave_recall / nq:.4f}", file=sys.stderr)
    print(f"插入耗时: {duration_insert:.4f} seconds", file=sys.stderr)


if __name__ == "__main__":
    main()