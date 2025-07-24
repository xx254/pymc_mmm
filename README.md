# PyMC Media Mix Modeling (MMM) Tutorial

这是一个使用 PyMC-Marketing 的因果媒体混合建模教程，包含交互式的因果DAG编辑器。

## 功能特性

### 🎯 因果DAG编辑器
- **可视化DAG构建**: 拖拽式节点和边的创建
- **多种节点类型**: Treatment (处理变量), Outcome (结果变量), Confounder (混淆变量), Unobserved (未观测变量)
- **实时DAG导出**: 支持Graphviz DOT格式导出
- **预设DAG模板**: 提供业务场景和简化版DAG模板

### 📊 数据支持
- **默认数据集**: 使用内置的 `data_mmm.csv` 数据集
- **自定义数据上传**: 支持拖拽上传CSV文件进行模型训练
- **智能数据映射**: 自动检测和映射常见的列名（date, target, x1, x2等）
- **数据格式验证**: 确保上传的数据符合MMM建模要求

### 🚀 模型训练
- **因果模型vs相关性模型**: 支持两种不同的建模方法
- **实时训练状态**: 显示模型训练进度和结果
- **模型评估指标**: R²、MAPE、MAE、RMSE等评估指标
- **收敛性诊断**: 自动检查模型收敛性

### 📈 结果可视化
- **拟合质量图**: 实际值vs预测值散点图
- **时间序列图**: 带置信区间的时间序列预测
- **残差分析**: 残差分布和残差图
- **媒体贡献分解**: X1和X2渠道的贡献恢复效果

## 安装和运行

### 环境要求
```bash
# Python 环境
python >= 3.9

# 核心依赖
pip install pymc pymc-marketing arviz pandas numpy matplotlib seaborn
pip install fastapi uvicorn python-multipart

# 前端环境 (Node.js)
npm install
```

### 启动服务

1. **启动后端API服务器**:
```bash
python api_server.py
```
API将在 `http://localhost:8000` 运行

2. **启动前端React应用**:
```bash
cd causal-dag-editor
npm start
```
前端将在 `http://localhost:3000` 运行

## 使用指南

### 1. 数据准备

#### 使用默认数据
- 选择 "Use default dataset (data_mmm.csv)" 选项
- 系统将使用内置的示例数据集进行模型训练

#### 上传自定义数据
1. 选择 "Upload custom CSV file" 选项
2. 拖拽CSV文件到上传区域，或点击浏览文件
3. 确保CSV文件包含以下列：

**推荐的数据格式**:
```csv
date_week,y,x1,x2,holiday_signal
2022-01-01,1500.5,245.2,198.7,0.1
2022-01-08,1620.3,267.8,210.4,0.0
...
```

**列名映射规则**:
- **日期列**: `date_week`, `date`, `time` 等
- **目标变量**: `y`, `target`, `sales`, `revenue`, `conversion` 等
- **营销渠道1**: `x1`, `social`, `facebook`, `social_media`, `channel1` 等
- **营销渠道2**: `x2`, `search`, `google`, `search_engine`, `channel2` 等
- **控制变量**: `holiday_signal`, `event_1`, `competitor` 等

### 2. DAG构建

1. **选择DAG类型**:
   - **Business Scenario**: 完整的业务场景DAG
   - **Simple DAG**: 简化版DAG
   - **Custom DAG**: 从空白画布开始自定义

2. **添加节点**:
   - 从左侧工具箱拖拽节点类型到画布
   - 双击节点可编辑标签

3. **连接节点**:
   - 点击源节点的连接点，拖拽到目标节点
   - 箭头表示因果关系方向

### 3. 模型训练

1. 确保DAG结构包含至少一个节点
2. 选择数据源（默认数据或自定义上传）
3. 点击 "🚀 Train Causal Model" 按钮
4. 等待训练完成，查看结果

### 4. 结果解读

#### 模型摘要
- **模型类型**: 因果模型 vs 相关性模型
- **DAG信息**: 节点数、边数、处理变量、结果变量
- **拟合质量**: R²分数、误差指标

#### 收敛性诊断
- **R-hat值**: 应接近1.0（< 1.1表示收敛良好）
- **有效样本量**: 应足够大（通常> 400）
- **分歧数**: 应为0或很小

## 技术架构

### 后端 (Python + FastAPI)
- **API服务器**: `api_server.py` - 处理HTTP请求和模型训练
- **MMM教程**: `causal_mmm_tutorial.py` - 核心建模逻辑
- **数据处理**: 自动数据预处理和格式转换

### 前端 (React + TypeScript)
- **DAG编辑器**: 基于ReactFlow的可视化编辑器
- **文件上传**: 拖拽式文件上传组件
- **结果展示**: Chart.js驱动的数据可视化

### 数据流
```
CSV文件 → 数据预处理 → DAG映射 → PyMC模型 → 训练结果 → 可视化展示
```

## 故障排除

### 常见问题

1. **模型训练失败**:
   - 检查数据格式是否正确
   - 确保数值列不包含缺失值
   - 验证DAG结构的合理性

2. **文件上传问题**:
   - 确保文件格式为CSV
   - 检查文件大小（限制10MB）
   - 验证CSV编码格式（推荐UTF-8）

3. **收敛性问题**:
   - 增加采样次数（draws）
   - 调整目标接受率（target_accept）
   - 检查数据质量和预处理

### 调试日志
后端API会输出详细的调试信息，包括：
- 数据加载状态
- 模型训练进度
- 错误详情和堆栈跟踪

## 贡献指南

欢迎提交Pull Request和Issue！

### 开发环境设置
```bash
# 克隆仓库
git clone <repository-url>
cd pymc_mmm

# 安装Python依赖
pip install -r requirements.txt

# 安装前端依赖
cd causal-dag-editor
npm install

# 启动开发环境
python api_server.py  # 后端
npm start            # 前端
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有问题或建议，请创建GitHub Issue或联系项目维护者。 