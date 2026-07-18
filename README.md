# bitcoin-rnn-price-regression
A deep learning pipeline for Bitcoin close-price regression using an RNN in PyTorch, with feature scaling, train-test split, and R² evaluation.

pandas: CSV loading and feature/target selection
scikit-learn: StandardScaler, train-test split, and r2_score metric
torch: Tensor conversion, custom Dataset/DataLoader, RNN model, training, and evaluation
