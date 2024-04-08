###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    X_mean = X.mean(axis=0)
    X = (X - X_mean) / (X.max(axis=0) - X.min(axis=0))

    y_mean = y.mean()
    y = (y - y_mean) / (y.max() - y.min())
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    m = X.shape[0]
    ones = np.ones(m)
    X = np.column_stack((ones, X))
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    m = len(y)
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using
    the training set. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    m = len(y)
    J_history = np.zeros(num_iters)

    for iter in range(num_iters):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors)
        theta = theta - sum_delta
        J_history[iter] = compute_cost(X, y, theta)

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop
    the learning process once the improvement of the loss value is smaller
    than 1e-8. This function is very similar to the gradient descent
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors)
        theta = theta - sum_delta
        J_history[iter] = compute_cost(X, y, theta)
        if iter > 0 and np.subtract(J_history[iter - 1], J_history[iter]) < 1e-8:
            break
    return theta, J_history[: iter + 1]


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using
    the training dataset. maintain a python dictionary with alpha as the
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [
        0.00001,
        0.00003,
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1,
        2,
        3,
    ]
    alpha_dict = {}  # {alpha_value: validation_loss}
    np.random.seed(42)
    random_theta = np.random.rand(X_train.shape[1])
    for alpha in alphas:
        theta, _ = efficient_gradient_descent(
            X_train, y_train, random_theta, alpha, iterations
        )
        cost = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = cost
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to
    select the most relevant features for a predictive model. The objective
    of this algorithm is to improve the model's performance by identifying
    and using only the most relevant features, potentially reducing overfitting,
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    remaining_features = list(range(X_train.shape[1]))
    while len(selected_features) < 5:
        best_cost = np.inf
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train_subset = apply_bias_trick(X_train[:, current_features])
            X_val_subset = apply_bias_trick(X_val[:, current_features])
            np.random.seed(42)
            random_theta = np.random.rand(X_train_subset.shape[1])
            theta, _ = efficient_gradient_descent(
                X_train_subset, y_train, random_theta, best_alpha, iterations
            )
            cost = compute_cost(X_val_subset, y_val, theta)
            if cost < best_cost:
                best_cost = cost
                best_feature = feature
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """
    df_poly = df.copy()
    # Create square features for each numeric column
    num_cols = df.select_dtypes(include=np.number).columns
    for col1 in num_cols:
        for col2 in num_cols:
            if col1 == col2:
                # Don't create duplicate features for the same column
                df_poly[f"{col1}^2"] = df[col1] ** 2
            elif col1 < col2:
                # Only create one feature per pair of columns to avoid redundancy
                df_poly[f"{col1}*{col2}"] = df[col1] * df[col2]
    return df_poly
