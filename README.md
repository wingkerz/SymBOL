# SymBOL
SymBOL: A Universal Symbolic Learner for Scientific Discovery Using Bayesian Optimization-Enhanced Large Language Models
🛠 安装指南
首先，请确保你的环境符合要求。根据提供的 YAML 文件安装相关依赖：

Bash
conda env create -f environment.yml
# 或者使用 pip
pip install -r requirements.txt
🤖 模型配置
在使用之前，需要配置 Embedding 模型和 LLM。

1. Embedding Model
请下载指定的 Embedding 模型至 model/ 文件夹。

在 main.py 中更新对应的模型路径。

2. Large Language Models (LLM)
本地开源模型：下载模型至 model/ 文件夹，并修改 use_localmodel.py 中的加载路径。

闭源模型 (API)：若使用 GPT 等闭源模型，请在 use_gpt.py 中配置你的 API_KEY 和 ENDPOINT。

📊 数据准备
项目分为两个主要任务场景：

1. 低维一般符号回归
数据集：采用 LSR-transformer 数据集。

测试数据：data/ 文件夹下已内置部分测试数据。

切换用例：通过修改 screen_pretrain_knowledge_eq.py 中的 case_name 来更换测试例子。

自定义数据：如需测试其他数据，请参照现有格式进行生成。

2. 高维网络动力学
数据生成：运行 generate_data.py 生成动力学数据。

配置参数：在 configs/ 目录下的配置文件中设置网络拓扑结构、积分时间步长及区间。

示例 (Lotka-Volterra)：

data/dataset_test3.csv 包含了 Lotka-Volterra 场景的方程。

运行生成脚本后，将自动产出对应的数据文件。

🚀 运行测试
配置完成后，运行以下命令测试高维网络动力学符号回归：

Bash
python main.py

