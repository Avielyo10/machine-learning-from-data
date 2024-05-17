import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.1,
            (1, 1): 0.6
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        # TODO - validate probability, sum is 1.01 !!!!
        self.X_Y_C = {
            (0, 0, 0): 0.08,
            (0, 0, 1): 0.02,
            (0, 1, 0): 0.14,
            (0, 1, 1): 0.07,
            (1, 0, 0): 0.14,
            (1, 0, 1): 0.07,
            (1, 1, 0): 0.245,
            (1, 1, 1): 0.245,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        
        # X and Y independent IFF P(X,Y) = P(X)P(Y)
        for key,value in X_Y.items():
            k_x, k_y = key
            if not np.isclose( value, X[k_x]*Y[k_y] ):
                return True

        return False

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        
        # X and Y conditionaly independent given C IFF P(X,Y|C)=P(X|C)P(Y|C)
        for key,value in X_Y_C.items():
            k_x, k_y, k_c = key

            # P(X,Y|C)= P(X,Y,C) / P(C)
            x_y_given_c = value / C[k_c]    # TODO - divide by 0 validation? (multiple places)
            
            # P(X|C) = P(X,C) / P(C)
            x_given_c = X_C[(k_x,k_c)] / C[k_c] 
            
            # P(Y|C) = P(Y,C) / P(C)
            y_given_c = Y_C[(k_y,k_c)] / C[k_c] 

            if not np.isclose( x_y_given_c, x_given_c * y_given_c ):
                return False

        return True
        

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf (probability mass function) value for instance k given the rate
    """
    log_p = np.log((rate ** k) * np.exp(-rate) / np.math.factorial(k))
    
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.zeros(len(rates))
    
    for idx, rate in enumerate(rates):
        # log(a*b) = log(a) + log(b)
        likelihoods[idx] = np.sum([poisson_log_pmf(k, rate) for k in samples])

    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    max_rate_ind = np.argmax(likelihoods)
    rate = rates[max_rate_ind]

    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.sum(samples) / len(samples)
    
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf (probability density function) according to the given mean and std for the given x.    
    """
    variance = np.square(std)
    e_power = np.square(x - mean) / (2 * variance)
    
    p = np.exp(-e_power) / np.sqrt(2 * np.pi * variance)
    
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """

        self.class_value = class_value 
        
        labels = dataset[:, -1]
        self.total_dataset = len(labels)

        # drop class/label/last column
        dataset = dataset[:, :-1] 

        # filter rows that match the class value  
        rows = labels == class_value
        self.sub_data = dataset[rows]

        # compue mean/std vectors (Naive Bayes - covariance matrix not required)
        self.mean_vec = self.sub_data.mean(axis=0)
        self.std_vec = self.sub_data.std(axis=0)


    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.sub_data) / self.total_dataset

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        probs = np.array([normal_pdf(val, self.mean_vec[idx], self.std_vec[idx]) for idx, val in enumerate(x)])
        likelihood = np.prod(probs)
        
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 1
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            pred = 0

        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    labels = test_set[:, -1]
    
    # drop class/label/last column
    test_set = test_set[:, :-1]

    success = 0 
    for idx, x in enumerate(test_set):
        if map_classifier.predict(x) == labels[idx]:
            success += 1 

    acc = success / len(labels)

    return acc


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    
    d = len(x)
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    delta_mean = x - mean

    # TODO - check matrix product - in particular transpose
    numerator = np.exp( -0.5 * delta_mean.transpose().dot(cov_inv.dot(delta_mean)) ) 
    denominator = np.sqrt( ((2 * np.pi) ** d) * cov_det) 
    pdf = numerator / denominator

    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        
        self.class_value = class_value 
        
        labels = dataset[:, -1]
        self.total_dataset = len(labels)

        # drop class/label/last column
        dataset = dataset[:, :-1] 

        # filter rows that match the class value  
        rows = labels == class_value
        self.sub_data = dataset[rows]

        # compue mean vector and covariance matrix
        self.mean_vec = self.sub_data.mean(axis=0)
        self.cov_mtx = np.cov(self.sub_data, rowvar=0)

        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.sub_data) / self.total_dataset

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """

        likelihood = multi_normal_pdf(x, self.mean_vec, self.cov_mtx)        
        
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        # TODO - documentation is wrong - should be prior instead of posterior ? (PIAZZA)
        pred = 1
        if self.ccd0.get_prior() > self.ccd1.get_prior():
            pred = 0

        return pred

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 1
        if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x):
            pred = 0

        return pred

# TODO - use?
EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        
        self.class_value = class_value 
        
        labels = dataset[:, -1]
        self.total_dataset = len(labels)

        # drop class/label/last column
        dataset = dataset[:, :-1] 

        # filter rows that match the class value  
        rows = labels == class_value
        self.sub_data = dataset[rows]

        # Count for each feature (column) unique values
        self.features_map = {}
        for feature in range(self.sub_data.shape[1]):
            unique_values, counts = np.unique(self.sub_data[:, feature], return_counts=True)
            self.features_map[feature] = {val: counts[idx] for idx, val in enumerate(unique_values)}

    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = len(self.sub_data) / self.total_dataset

        return prior
    
    def get_laplace_likelihood(self, x_i, v_i):
        feature_imap = self.features_map.get(x_i, {})     
        
        # number of instances with the feature value
        n_ij = feature_imap.get(v_i, 0)

        # number of possible feature values
        n_j = len(feature_imap.keys())  

        likelihood = (n_ij + 1) / (self.total_dataset + n_j)
        
        return likelihood


    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        
        probs = np.array([self.get_laplace_likelihood(idx, val) for idx, val in enumerate(x)])
        likelihood = np.prod(probs)

        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_prior() * self.get_instance_likelihood(x)

        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1


    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = 1
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            pred = 0

        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        labels = test_set[:, -1]
    
        # drop class/label/last column
        test_set = test_set[:, :-1]

        success = 0 
        for idx, x in enumerate(test_set):
            if self.predict(x) == labels[idx]:
                success += 1 

        acc = success / len(labels)

        return acc


