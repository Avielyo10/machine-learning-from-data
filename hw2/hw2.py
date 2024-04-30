import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {
    1: {0.5: 0.45, 0.25: 1.32, 0.1: 2.71, 0.05: 3.84, 0.0001: 100000},
    2: {0.5: 1.39, 0.25: 2.77, 0.1: 4.60, 0.05: 5.99, 0.0001: 100000},
    3: {0.5: 2.37, 0.25: 4.11, 0.1: 6.25, 0.05: 7.82, 0.0001: 100000},
    4: {0.5: 3.36, 0.25: 5.38, 0.1: 7.78, 0.05: 9.49, 0.0001: 100000},
    5: {0.5: 4.35, 0.25: 6.63, 0.1: 9.24, 0.05: 11.07, 0.0001: 100000},
    6: {0.5: 5.35, 0.25: 7.84, 0.1: 10.64, 0.05: 12.59, 0.0001: 100000},
    7: {0.5: 6.35, 0.25: 9.04, 0.1: 12.01, 0.05: 14.07, 0.0001: 100000},
    8: {0.5: 7.34, 0.25: 10.22, 0.1: 13.36, 0.05: 15.51, 0.0001: 100000},
    9: {0.5: 8.34, 0.25: 11.39, 0.1: 14.68, 0.05: 16.92, 0.0001: 100000},
    10: {0.5: 9.34, 0.25: 12.55, 0.1: 15.99, 0.05: 18.31, 0.0001: 100000},
    11: {0.5: 10.34, 0.25: 13.7, 0.1: 17.27, 0.05: 19.68, 0.0001: 100000},
}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - np.sum(probabilities**2)
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


class DecisionNode:

    def __init__(
        self,
        data,
        impurity_func,
        feature=-1,
        depth=0,
        chi=1,
        max_depth=1000,
        gain_ratio=False,
    ):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        labels = self.data[:, -1]
        if labels.size == 0:
            return None  # Handle case with no data in the node
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """
        if (
            self.terminal or self.feature == -1
        ):  # Terminal or no feature used for splitting
            self.feature_importance = 0
            return

        # Impurity of the current node
        n_samples = self.data.shape[0]  # Number of samples at the current node
        node_impurity = self.impurity_func(self.data)

        # Calculate weighted impurity for the current node
        weighted_impurity_current = (n_samples / n_total_sample) * node_impurity

        # Calculate the sum of weighted impurities for each child node
        weighted_impurities_children = np.sum(
            [
                (child.data.shape[0] / n_total_sample) * self.impurity_func(child.data)
                for child in self.children
            ]
        )

        # Feature importance calculation
        self.feature_importance = (
            weighted_impurity_current - weighted_impurities_children
        )

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        # Splitting the data by feature values
        unique_values, indices = np.unique(self.data[:, feature], return_inverse=True)
        groups = {
            val: self.data[indices == idx] for idx, val in enumerate(unique_values)
        }
        impurity_func = calc_entropy if self.gain_ratio else self.impurity_func

        # Calculate impurity before the split
        impurity_before = impurity_func(self.data)

        # Calculate weighted impurity after the split
        n_samples = len(self.data)
        weighted_impurity_after = sum(
            (len(group) / n_samples) * impurity_func(group)
            for group in groups.values()
        )

        # Goodness of Split
        goodness = impurity_before - weighted_impurity_after

        if self.gain_ratio:
            # Calculate split information
            split_info = -sum(
                (len(group) / n_samples) * np.log2(len(group) / n_samples)
                for group in groups.values()
                if len(group) > 0  # Avoid log2(0) which is undefined
            )
            # Gain Ratio
            if split_info > 0:
                goodness = goodness / split_info
            else:
                goodness = 0  # To handle cases where split_info is zero, avoiding division by zero

        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        if self.depth >= self.max_depth or len(self.data) == 0:
            self.terminal = True
            return  # Stop splitting if max depth is reached or no data is present

        # Select the best feature based on the impurity reduction
        best_goodness = -np.inf
        best_feature = None
        best_splits = {}

        for feature in range(self.data.shape[1] - 1):  # Assume last column is the label
            goodness, splits = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_splits = splits

        # Check if splitting improves or possible pruning conditions
        if best_goodness <= 0 or not best_splits:
            self.terminal = True
            return  # No improvement in impurity or no splits available

        # Chi square pruning
        if self.chi < 1:
            labels = self.data[:, -1]
            unique_labels, counts = np.unique(labels, return_counts=True)
            degree_of_freedom = (len(best_splits.keys()) - 1) * (len(unique_labels) - 1)
            
            # Prepare a chi square array to accumulate the chi square value
            chi_square_val = 0

            # Vectorizing the chi square calculation
            for subset in best_splits.values():
                sub_labels = subset[:, -1]
                # Get counts of each label in subset
                label_counts = np.array([np.sum(sub_labels == label) for label in unique_labels])
                # Calculate expected counts
                expected_counts = len(subset) * (counts / counts.sum())
                # Vectorized chi square calculation for this subset
                chi_square_val += np.sum(((label_counts - expected_counts) ** 2) / expected_counts)

            # Pruning condition check using chi_table
            if chi_square_val <= chi_table[degree_of_freedom][self.chi]:
                self.terminal = True
                return

        # Create child nodes for each split
        self.feature = best_feature
        for val, subset in best_splits.items():
            if len(subset) > 0:  # Only create a child if there are data points
                child_node = DecisionNode(
                    data=subset,
                    impurity_func=self.impurity_func,
                    depth=self.depth + 1,
                    chi=self.chi,
                    max_depth=self.max_depth,
                    gain_ratio=self.gain_ratio,
                )
                self.add_child(child_node, val)


class DecisionTree:
    def __init__(
        self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False
    ):
        self.data = data  # the relevant data for the tree
        self.impurity_func = (
            impurity_func  # the impurity function to be used in the tree
        )
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root = None  # the root node of the tree
        self.tree_depth = 0  # the depth of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio,
        )
        queue = [root]
        n_total = len(self.data)

        while queue:
            node = queue.pop(0)
            node.split()
            node.calc_feature_importance(n_total)
            self.tree_depth = max(self.tree_depth, node.depth)  # Update tree depth
            queue.extend([child for child in node.children if not child.terminal])
        self.root = root

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        node = self.root
        while not node.terminal:
            feature_value = instance[node.feature]
            child_idx = np.where(np.array(node.children_values) == feature_value)[0]
            if len(child_idx) == 0:
                return node.pred
            node = node.children[child_idx[0]]
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        success = 0
        for index in range(0, dataset.shape[0]):
            predict_index = self.predict(dataset[index])
            if predict_index == dataset[index, -1]:
                success += 1
        return success / dataset.shape[0]

    def depth(self):
        """
        Calculate the depth of the tree

        Output: the depth of the tree
        """
        return self.tree_depth


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(X_train, calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    # add no-pruning first - p-value cut-off is 1
    tree = DecisionTree(data=X_train, impurity_func=calc_entropy, gain_ratio=True)
    tree.build_tree()
    chi_training_acc.append(tree.calc_accuracy(X_train))
    chi_validation_acc.append(tree.calc_accuracy(X_test))
    depth.append(tree.depth())

    for p_value in [0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(
            data=X_train, impurity_func=calc_entropy, gain_ratio=True, chi=p_value
        )
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))
        depth.append(tree.depth())

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """
    n_nodes = 1
    if node.children:
        for child in node.children:
            n_nodes += count_nodes(child)
    return n_nodes
