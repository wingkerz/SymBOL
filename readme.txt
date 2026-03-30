1.Install dependencies according to the yaml file.
2.Load models
  Embedding model: 
    Download the embedding model into the model folder, and update the corresponding path in main.py.
  LLM models: 
    For locally deployed open-source LLMs, download the model into the model folder and modify the model loading path in use_localmodel.py. For closed-source LLMs, update the API endpoint and key in use_gpt.py.
3.Load data
  Using the Lotka-Volterra scenario from network dynamics as an example, the data is stored in data/test_data_on_Lotka_Volterra.pickle.
The data is loaded and evaluated via evaluator.py.
4.Run main.py.
  
  