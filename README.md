# SymBOL
SymBOL: A Universal Symbolic Learner for Scientific Discovery Using Bayesian Optimization-Enhanced Large Language Models
# 🚀 快速开始

## 1. 环境配置

首先，请确保你已安装必要的依赖项。项目提供了 `yaml` 环境配置文件：

```bash
# 使用 conda 创建环境
conda env create -f environment.yml

# 激活环境
conda activate [你的环境名称]
```

---

## 2. 模型准备

在使用之前，需要配置 **Embedding 模型** 和 **大语言模型（LLM）**。

### Embedding 模型

1. 下载预训练的 Embedding 模型。
2. 将模型文件存放到 `model/` 文件夹中。
3. 在 `main.py` 中更新模型对应的存放路径。

### LLM 模型配置

项目支持 **本地部署模型** 及 **在线 API**：

#### 本地开源模型

1. 将模型文件下载至 `model/` 文件夹。
2. 在 `use_localmodel.py` 中修改模型加载路径。

#### 闭源模型（API）

若使用 GPT 等在线模型：

1. 在 `use_gpt.py` 中更新 API Endpoint。
2. 填入你的 API Key。

---

# 📊 数据准备

项目针对不同的应用场景提供了两种数据处理方式：

## A. 低维一般符号回归

项目采用 **LSR-transformer** 数据集。

* **测试数据**：`data/` 文件夹下已内置生成的测试数据
* **切换用例**：修改 `screen_pretrain_knowledge_eq.py` 中的 `case_name` 即可更换测试例子
* **自定义数据**：如需测试其他数据集，请参考现有格式生成相应实验数据

---

## B. 高维网络动力学

以 **Lotka-Volterra** 情景为例，通过模拟生成网络动力学数据。

### 数据生成

运行以下脚本生成数据：

```bash
python generate_data.py
```

### 配置修改

在 `configs/` 文件夹下的配置文件中，可以自定义：

* 网络拓扑结构
* 积分时间步长
* 时间区间

### 方程参考

`data/dataset_test3.csv` 中包含了 **Lotka-Volterra** 情景对应的具体方程。

---

# 💻 运行与测试

完成模型配置和数据准备后，运行主程序进行高维网络动力学符号回归测试：

```bash
python main.py
```

---

# 📁 项目结构（建议）

```
project/
│
├── configs/
├── data/
├── model/
├── generate_data.py
├── main.py
├── use_gpt.py
├── use_localmodel.py
└── screen_pretrain_knowledge_eq.py
```

