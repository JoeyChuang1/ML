import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1234)

class Node:
    def __init__(self, data_indices, parent):
        self.data_indices = data_indices
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        if parent:
            self.depth = parent.depth + 1
            self.num_classes = parent.num_classes
            self.data = parent.data
            self.labels = parent.labels
            class_prob = np.bincount(self.labels[data_indices], minlength=self.num_classes)
            self.class_prob = class_prob / np.sum(class_prob)
def greedy_test(node, cost_fn):

    best_cost = np.inf
    best_feature, best_value = None, None
    num_instances, num_features = node.data.shape

    data_sorted = np.sort(node.data[node.data_indices],axis=0)
    test_candidates = (data_sorted[1:] + data_sorted[:-1]) / 2.
    for f in range(num_features):

        data_f = node.data[node.data_indices, f]
        for test in test_candidates[:,f]:

            left_indices = node.data_indices[data_f <= test]
            right_indices = node.data_indices[data_f > test]

            if len(left_indices) == 0 or len(right_indices) == 0:
                continue

            left_cost = cost_fn(node.labels[left_indices])
            right_cost = cost_fn(node.labels[right_indices])
            num_left, num_right = left_indices.shape[0], right_indices.shape[0]

            cost = (num_left * left_cost + num_right * right_cost)/num_instances

            if cost < best_cost:
                best_cost = cost
                best_feature = f
                best_value = test
    return best_cost, best_feature, best_value
def cost_misclassification(labels):
    counts = np.bincount(labels)
    class_probs = counts / np.sum(counts)

    return 1 - np.max(class_probs)


def cost_entropy(labels):
    class_probs = np.bincount(labels) / len(labels)
    class_probs = class_probs[class_probs > 0]
    return -np.sum(class_probs * np.log(class_probs))


def cost_gini_index(labels):
    class_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(np.square(class_probs))


class DecisionTree:
    def __init__(self, num_classes=None, max_depth=3, cost_fn=cost_gini_index, min_leaf_instances=1):
        self.max_depth = max_depth
        self.root = None
        self.cost_fn = cost_fn
        self.num_classes = num_classes
        self.min_leaf_instances = min_leaf_instances

    def fit(self, data, labels):
        pass

    def predict(self, data_test):
        pass

def fit(self, data, labels):
    self.data = data
    self.labels = labels
    if self.num_classes is None:
        self.num_classes = np.max(labels) + 1
    #below are initialization of the root of the decision tree
    self.root = Node(np.arange(data.shape[0]), None)
    self.root.data = data
    self.root.labels = labels
    self.root.num_classes = self.num_classes
    self.root.depth = 0
    #to recursively build the rest of the tree
    self._fit_tree(self.root)
    return self

def _fit_tree(self, node):

    if node.depth == self.max_depth or len(node.data_indices) <= self.min_leaf_instances:
        return

    cost, split_feature, split_value = greedy_test(node, self.cost_fn)

    if np.isinf(cost):
        return

    test = node.data[node.data_indices,split_feature] <= split_value
    #store the split feature and value of the node
    node.split_feature = split_feature
    node.split_value = split_value
    #define new nodes which are going to be the left and right child of the present node
    left = Node(node.data_indices[test], node)
    right = Node(node.data_indices[np.logical_not(test)], node)
    #recursive call to the _fit_tree()
    self._fit_tree(left)
    self._fit_tree(right)
    #assign the left and right child to present child
    node.left = left
    node.right = right

DecisionTree.fit = fit
DecisionTree._fit_tree = _fit_tree
def predict(self, data_test):
    class_probs = np.zeros((data_test.shape[0], self.num_classes))
    for n, x in enumerate(data_test):
        node = self.root
        #loop along the dept of the tree looking region where the present data sample fall in based on the split feature and value
        while node.left:
            if x[node.split_feature] <= node.split_value:
                node = node.left
            else:
                node = node.right
        #the loop terminates when you reach a leaf of the tree and the class probability of that node is taken for prediction
        class_probs[n,:] = node.class_prob
    return class_probs

DecisionTree.predict = predict
dataset=pd.read_csv("/data/newdata2.csv", error_bad_lines=False)

#dataset=dataset.astype(int)
# #x=np.array(dataset[['B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','T',]])
x = np.array(dataset[['R','Q']])
#print(x)
#
dataset=dataset.values
# #x=dataset[:,:2]
y = dataset[:,-1]
y=y.astype(np.int32)
(num_instances, num_features), num_classes = x.shape, np.max(y)+1
inds = np.random.permutation(num_instances)
#train-test split)
x_train, y_train = x[inds[:1000]], y[inds[:1000]]
x_test, y_test = x[inds[148:]], y[inds[148:]]
tree = DecisionTree(max_depth=35)
probs_test = tree.fit(x_train, y_train).predict(x_test)
y_pred = np.argmax(probs_test,1)
accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
print(f'accuracy is {accuracy*100:.1f}.')


#visualization
# correct = y_test == y_pred
# incorrect = np.logical_not(correct)
# plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
# plt.scatter(x_test[correct,0], x_test[correct,1], marker='.', c=y_pred[correct], label='correct')
# plt.scatter(x_test[incorrect,0], x_test[incorrect,1], marker='x', c=y_test[incorrect], label='misclassified')
# plt.legend()
# plt.show()
#decision boundry
# x0v = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 200)
# x1v = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 200)
# x0,x1 = np.meshgrid(x0v, x1v)
# x_all = np.vstack((x0.ravel(),x1.ravel())).T
#
# model = DecisionTree(max_depth=4)
# y_train_prob = np.zeros((y_train.shape[0], num_classes))
# y_train_prob[np.arange(y_train.shape[0]), y_train] = 1
# y_prob_all = model.fit(x_train, y_train).predict(x_all)
# plt.scatter(x_train[:,0], x_train[:,1], c=y_train_prob, marker='o', alpha=1)
# plt.scatter(x_all[:,0], x_all[:,1], c=y_prob_all, marker='.', alpha=.01)
# plt.ylabel('sepal length')
# plt.xlabel('sepal width')
# plt.show()


# #decision boundry
# x0v = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 1000)
# x1v = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 1000)
# x0,x1 = np.meshgrid(x0v, x1v)
# x_all = np.vstack((x0.ravel(),x1.ravel())).T
# #
# model = DecisionTree(max_depth=35)
# y_train_prob = np.zeros((y_train.shape[0], num_classes))
# y_train_prob[np.arange(y_train.shape[0]), y_train] = 1
# y_prob_all = model.fit(x_train, y_train).predict(x_all)
# plt.scatter(x_train[:,0], x_train[:,1], c=y_train_prob, marker='o', alpha=1)
# plt.scatter(x_all[:,0], x_all[:,1], c=y_prob_all, marker='.', alpha=.01)
# plt.ylabel('sepal length')
# plt.xlabel('sepal width')
# plt.show()

# tree = DecisionTree(max_depth=20)
# probs_test = tree.fit(x_train, y_train).predict(x_test)
# y_pred = np.argmax(probs_test,1)
#
#
# y_train_prob = np.zeros((y_train.shape[0], num_classes))
# y_train_prob[np.arange(y_train.shape[0]), y_train] = 1
# y_prob_all = tree.fit(x_train, y_train).predict(x_all)
# plt.scatter(x_train[:,0], x_train[:,1], c=y_train_prob, marker='o', alpha=1)
# plt.scatter(x_all[:,0], x_all[:,1], c=y_prob_all, marker='.', alpha=.01)
# plt.ylabel('ALBUMIN')
# plt.xlabel('ASCITES')
# plt.show()




