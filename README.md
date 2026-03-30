# SymBOL
SymBOL: A Universal Symbolic Learner for Scientific Discovery Using Bayesian Optimization-Enhanced Large Language Models
🛠 Installation
Install the required dependencies using the provided environment configuration file:

Bash
# Using Conda
conda env create -f environment.yml

# Or using pip if a requirements.txt is provided
pip install -r requirements.txt
🤖 Model Configuration
1. Embedding Model
Download your chosen embedding model into the model/ directory.

Update the model storage path in main.py to point to your local file.

2. Large Language Models (LLM)
Local Open-Source LLMs: Download the model weights into the model/ directory and modify the loading path in use_localmodel.py.

Closed-Source LLMs (API): Update your API_ENDPOINT and API_KEY in use_gpt.py.

📊 Data Preparation
The project supports two primary scenarios:

1. Low-Dimensional General Symbolic Regression
Dataset: Utilizes the LSR-transformer dataset.

Testing: Pre-generated test data is available in the data/ folder.

Configuration: Change the case_name in screen_pretrain_knowledge_eq.py to switch between different test cases.

Custom Data: If using external datasets, ensure they follow the project's formatting requirements.

2. High-Dimensional Network Dynamics
Generation: Run generate_data.py to create network dynamics data.

Settings: Configure the network topology, integration time steps, and intervals in the files located under the configs/ directory.

Example (Lotka-Volterra):

The repository uses the Lotka-Volterra scenario as a default example.

data/dataset_test3.csv contains the relevant equations; running the script will generate the corresponding dynamic data.

🚀 Usage
To test the High-Dimensional Network Dynamics Symbolic Regression, run the following command:

Bash
python main.py
