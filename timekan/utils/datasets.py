
import torch

def mackey_glass(length=1200, tau=17, a=0.2, b=0.1, n=10, x0=1.2, window_size=20, train_size=1000):
    """
    Generate training and testing datasets for the Mackey-Glass time series.
    
    Parameters:
        length (int): Total number of points in the time series (default: 1200)
        tau (int): Delay parameter (default: 17)
        a (float): Equation parameter 'a' (default: 0.2)
        b (float): Equation parameter 'b' (default: 0.1)
        n (int): Equation parameter 'n' (default: 10)
        x0 (float): Initial condition (default: 1.2)
        window_size (int): Number of past points to predict the next point (default: 20)
        train_size (int): Number of points for training (default: 1000)
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test)
            - x_train (torch.Tensor): Training inputs, shape [980, 20, 1]
            - y_train (torch.Tensor): Training targets, shape [980]
            - x_test (torch.Tensor): Testing inputs, shape [180, 20, 1]
            - y_test (torch.Tensor): Testing targets, shape [180]
    """
    x = [x0]
    for t in range(1, length):
        x_delay = x0 if t < tau else x[t - tau]
        dx = -b * x[t - 1] + a * x_delay / (1 + x_delay**n)
        x.append(x[t - 1] + dx)
    
    train_series = x[:train_size]
    min_val = min(train_series)
    max_val = max(train_series)
    x_scaled = [(val - min_val) / (max_val - min_val) for val in x]
    
    x_train = [x_scaled[t:t + window_size] for t in range(train_size - window_size)]
    y_train = [x_scaled[t + window_size] for t in range(train_size - window_size)]
    
    test_start = train_size
    test_end = length - window_size
    x_test = [x_scaled[t:t + window_size] for t in range(test_start, test_end)]
    y_test = [x_scaled[t + window_size] for t in range(test_start, test_end)]
    
    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    return x_train, y_train, x_test, y_test