# SymBOL
SymBOL: A Universal Symbolic Learner for Scientific Discovery Using Bayesian Optimization-Enhanced Large Language Models
#  Quick Start

## 1. Environment Setup

First, make sure you have installed the required dependencies. The project provides a `yaml` environment configuration file:

```bash
# Create environment using conda
conda env create -f environment.yml

# Activate the environment
conda activate [your_environment_name]
```

---

## 2. Model Preparation

Before running the project, you need to configure both the **Embedding model** and the **LLM**.

### Embedding Model

1. Download the pretrained embedding model.
2. Place the model files into the `model/` directory.
3. Update the model path in `main.py` and `use_localmodel.py`.

### LLM Configuration

The project supports both **local deployment** and **online API-based models**:

#### Local Open-source Models

1. Download the model files into the `model/` directory.
2. Modify the model loading path in `use_localmodel.py`.

#### Closed-source Models (API)

If you use online models such as GPT:

1. Update the API Endpoint in `use_gpt.py`.
2. Provide your API Key.
3. Replace all occurrences of `use_localmodel` with `use_gpt` in the functions **`gen_use_llm_separate`**, **`suggest_use_llm`**, as well as in the main function.

Modify the `prompts_path` in the main function to your local path, for example:
`prompts_path = "/home/SymBOL/prompts/SR/gen_prompts_2.txt"`


---

# Data Preparation

The project provides two data processing pipelines for different application scenarios:

## A. Low-dimensional Symbolic Regression

The project uses the **LSR-transformer dataset** (https://huggingface.co/datasets/nnheui/llm-srbench).

* **Train data**: Pre-generated train datasets are available in the `data/` directory.
* **Switch test cases**: Modify the `case_name` in `screen_pretrain_knowledge_eq.py` to select different examples.
* **Custom datasets**: To test on other datasets, follow the existing format to generate corresponding experimental data.

---

## B. High-dimensional Network Dynamics Symbolic Regression

Taking the **Lotka–Volterra** scenario as an example, network dynamic data is generated via simulation.

### Data Generation

Run the following script to generate data:

```bash
python generate_data.py
```

### Configuration

In the configuration files located in the `configs/Simulation_config.yaml` directory, you can customize:

* Network topology structure
* Integration time steps
* Time range

### Reference Equations

The file `data/dataset_test3.csv` contains the equations corresponding to the **Lotka–Volterra** scenario.

---

# Running Experiments

After configuring the models and preparing the data, run the main script to perform symbolic regression on high-dimensional network dynamics:

```bash
python main_high_SR.py
The low-dimensional symbolic regression task can be executed using main_low_SR.py.
In the prompt, the Operators list specifies the operators that may appear in the generated expressions.
```

---




