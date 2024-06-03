# An Empirical Analysis of Adversarial Attacks in Federated Learning

In this project, we experimentally analyze the susceptibility of selected Federated
Learning (FL) systems to the presence of adversarial clients. We find that temporal
attacks significantly affect model performance in FL, especially when the adversaries are
active throughout and during the ending rounds of the FL process. Machine Learning
models like Multinominal Logistic Regression, Support Vector Classifier (SVC), Neural
Network models like Multilayer Perceptron (MLP), Convolution Neural Network
(CNN), Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM) and
tree-based machine learning models like Random Forest and XGBoost were considered.
These results highlight the effectiveness of temporal attacks and the need to develop
strategies to make the FL process more robust. We also explore defense mechanisms,
including outlier detection in the aggregation algorithm.

Technologies: 
1. Python 
2. Flower A Friendly Federated Learning Framework (https://flower.ai/)
2. PyTorch 
3. scikit-learn

Files: 
1. main.py: To run an experiment 
2. adverseries-experiments.ipynb: To run all experiments for a given attack 
3. visualization.ipynb: To visualize the result files generated from the experiments

Folders: 
1. runs0: Experiments without defense mechanism 
2. runs2: Experiments with outlier detection 