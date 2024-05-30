import numpy as np
import pandas as pd


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values.

    Returns:
    - The Pearson correlation coefficient between the two columns.
    """
    if len(x) != len(y):
        raise ValueError("The input arrays must have the same length.")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("The input arrays must not be empty.")

    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate deviations from the mean
    x_deviation = x - mean_x
    y_deviation = y - mean_y

    # Calculate numerator
    numerator = np.sum(x_deviation * y_deviation)

    # Calculate denominator
    x_deviation_squared = np.sum(x_deviation**2)
    y_deviation_squared = np.sum(y_deviation**2)
    denominator = np.sqrt(x_deviation_squared * y_deviation_squared)

    if denominator == 0:
        return 0  # Avoid division by zero

    # Calculate Pearson correlation coefficient
    r = numerator / denominator

    return r


def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).
    """
    feature_correlations = {}

    # Calculate Pearson correlation for each feature
    for feature_name in X.columns:
        # check if the feature is numeric
        if not np.issubdtype(X[feature_name].dtype, np.number):
            continue
        correlation = pearson_correlation(X[feature_name].values, y)
        feature_correlations[feature_name] = abs(correlation)

    # Sort features based on absolute correlation values
    sorted_features = sorted(
        feature_correlations.items(), key=lambda item: item[1], reverse=True
    )

    # Select top n features
    best_features = [feature for feature, correlation in sorted_features[:n_features]]

    return best_features


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        m, n = X.shape
        self.theta = np.random.rand(n)
        prev_cost = None

        for _ in range(self.n_iter):
            h = self.sigmoid(np.dot(X, self.theta))

            cost = -(1 / m) * np.sum(y * np.log(h) + ((1 - y) * np.log(1 - h)))

            if prev_cost is not None and prev_cost - cost < self.eps:
                break

            prev_cost = cost
            self.Js.append(cost)
            self.thetas.append(self.theta)
            error = h - y
            gradient = np.dot(X.T, error)
            self.theta -= self.eta * gradient

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = np.insert(X, 0, 1, axis=1)  # add bias
        predictions = self.sigmoid(np.dot(X, self.theta))
        return np.where(predictions > 0.5, 1, 0)

    @staticmethod
    def sigmoid(z):
        """
        Compute the sigmoid of z
        """
        return 1 / (1 + np.exp(-z))


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """
    # Set random seed
    np.random.seed(random_state)

    # Shuffle the data
    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    # Split the data into folds
    fold_size = len(X) // folds
    fold_accuracies = []

    for fold in range(folds):
        start = fold * fold_size
        end = start + fold_size if fold != folds - 1 else len(X)  # Handle the last fold

        X_val = X_shuffled[start:end]
        y_val = y_shuffled[start:end]

        X_train = np.concatenate([X_shuffled[:start], X_shuffled[end:]])
        y_train = np.concatenate([y_shuffled[:start], y_shuffled[end:]])

        # Initialize the model
        model = algo
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred = model.predict(X_val)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_val)
        fold_accuracies.append(accuracy)

    # Calculate average accuracy for current eta and eps
    cv_accuracy = np.mean(fold_accuracies)

    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.
    """
    # Calculate the normal distribution pdf
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    return pdf


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        n = data.shape[0]
        self.weights = np.full(self.k, 1 / self.k)
        self.mus = np.random.rand(self.k)
        self.sigmas = np.random.rand(self.k)
        self.responsibilities = np.zeros((n, self.k))
        self.costs = []

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        for i in range(self.k):
            self.responsibilities[:, i] = self.weights[i] * np.prod(
                norm_pdf(data, self.mus[i], self.sigmas[i]), axis=1
            )
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        n = data.shape[0]
        self.weights = np.sum(self.responsibilities, axis=0) / n
        self.mus = np.sum(self.responsibilities * data, axis=0) * (
            1 / (self.weights * n)
        )
        self.sigmas = np.sqrt(
            np.sum(self.responsibilities * (data - self.mus) ** 2, axis=0)
            * (1 / (self.weights * n))
        )

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        prev_cost = None

        for _ in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            cost = self.compute_cost(data)
            self.costs.append(cost)

            if prev_cost is not None and abs(prev_cost - cost) < self.eps:
                break

            prev_cost = cost

    def compute_cost(self, data):
        """
        Compute the -log likelihood cost for the current parameters
        """
        return np.sum(-np.log(self.weights * norm_pdf(data, self.mus, self.sigmas)))

    def get_dist_params(self):
        """
        Return the distribution params
        """
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.
    """
    pdf = np.zeros(data.shape[0])
    for weight, mu, sigma in zip(weights, mus, sigmas):
        pdf += weight * np.prod(norm_pdf(data, mu, sigma), axis=1)
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.classes = None
        self.gmms = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        self.classes, counts = np.unique(y, return_counts=True)
        self.prior = counts / len(y)
        self.num_classes = len(self.classes)
        self.num_features = X.shape[1]

        for i in range(self.num_classes):
            self.gmms[self.classes[i]] = []
            class_data = X[y == self.classes[i]]
            for j in range(self.num_features):
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(class_data[:, j : j + 1])
                weights, mus, sigmas = em.get_dist_params()
                self.gmms[self.classes[i]].append((weights, mus, sigmas))

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        predictions = np.zeros([self.num_classes, X.shape[0]])
        for i in range(self.num_classes):
            likelihood = 1
            for j in range(self.num_features):
                prob = gmm_pdf(X[:, j : j + 1], *self.gmms[self.classes[i]][j])
                likelihood *= prob
            predictions[i] = self.prior[i] * likelihood
        return self.classes[np.argmax(predictions, axis=0)].reshape(-1, 1)


def accuracy(labels, predictions):
    """
    Calculate the accuracy of the model.

    Parameters
    ----------
    labels : array-like, shape = [n_examples]
      True labels.
    predictions : array-like, shape = [n_examples]
      Predicted labels.

    Returns the accuracy of the model.
    """
    return np.sum(predictions == labels) / len(predictions)


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    """
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    """

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################

    # Logistic Regression
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps, random_state=42)
    lor.fit(x_train, y_train)
    y_pred = lor.predict(x_train)
    lor_train_acc = accuracy(y_train, y_pred)
    y_pred = lor.predict(x_test)
    lor_test_acc = accuracy(y_test, y_pred)

    # Naive Bayes
    naive_bayes = NaiveBayesGaussian(k=k, random_state=42)
    naive_bayes.fit(x_train, y_train)
    y_pred = naive_bayes.predict(x_train)
    bayes_train_acc = accuracy(y_train.reshape(-1, 1), y_pred)
    y_pred = naive_bayes.predict(x_test)
    bayes_test_acc = accuracy(y_test.reshape(-1, 1), y_pred)

    ###########################################################################
    return {
        "lor_train_acc": lor_train_acc,
        "lor_test_acc": lor_test_acc,
        "bayes_train_acc": bayes_train_acc,
        "bayes_test_acc": bayes_test_acc,
        "models": {
            "lor": lor,
            "naive_bayes": naive_bayes,
        },
    }


def generate_datasets():
    from scipy.stats import multivariate_normal as mvn

    """
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    """
    dataset_a_features, dataset_a_labels = generate_dataset_a(mvn)
    dataset_b_features, dataset_b_labels = generate_dataset_b(mvn)

    return {
        "dataset_a_features": dataset_a_features,
        "dataset_a_labels": dataset_a_labels,
        "dataset_b_features": dataset_b_features,
        "dataset_b_labels": dataset_b_labels,
    }


def generate_dataset_a(mvn):
    np.random.seed(42)  # For reproducibility

    # Define mean vectors for the Gaussian distributions
    mean1 = [5, 10, -8]
    mean2 = [-6, -7, 4]
    mean3 = [-4, 6, 6]
    mean4 = [4, -6, -8]

    # Define covariance matrices
    cov = 4 * np.identity(3)  # Assuming identical covariance for simplicity

    # Generate samples
    dist1 = mvn.rvs(mean1, cov, size=500)
    dist2 = mvn.rvs(mean2, cov, size=500)
    dist3 = mvn.rvs(mean3, cov, size=500)
    dist4 = mvn.rvs(mean4, cov, size=500)

    # Create labels
    labels_class0 = np.zeros((1000, 1))
    labels_class1 = np.ones((1000, 1))
    dataset_a_labels = np.concatenate((labels_class0, labels_class1)).flatten()

    # Combine the data
    data_class0 = np.vstack((dist1, dist2))
    data_class1 = np.vstack((dist3, dist4))

    data_class0 = np.column_stack((data_class0, labels_class0))
    data_class1 = np.column_stack((data_class1, labels_class1))

    dataset_a_features = np.vstack((data_class0, data_class1))

    return dataset_a_features, dataset_a_labels


def generate_dataset_b(mvn):
    np.random.seed(42)  # For reproducibility

    # Generate linear data with Gaussian noise
    x1 = mvn.rvs(1, 1, 1000)
    x2 = 2 * x1 + mvn.rvs(0, 1.5, 1000)
    x3 = 3 * x2 + mvn.rvs(0, 2, 1000)

    labels = np.concatenate((np.zeros(500), np.ones(500)))
    x1[labels == 0] += 5  # Shift class 0 to make it linearly separable

    dataset_b_features = np.column_stack((x1, x2, x3))
    dataset_b_labels = labels

    return dataset_b_features, dataset_b_labels
