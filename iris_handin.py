import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

class LinearClassifier():
    def __init__(self, alpha = 0.005, num_classes = 3):
        self.num_classes = num_classes
        self.alpha = alpha
        self.losses, self.train_accuracies = [], []
        self.threshold = 0.000001
        self.num_iter = 0
        self.max_iter = 10000
        self.W = None
        self.m_samples, self.n_features = 0,0 
       
    def sigmoid(self, X):
        return np.array(1/(1+ np.exp(-X)))
    
    def one_hot(self, y, num_classes):
        onehot = np.zeros((y.shape[0], num_classes))
        onehot[np.arange(y.shape[0]), y] = 1
        return onehot
    
    def MSE_grad_W(self, G, T, X):
        return ((G - T) * G * (1 - G)) @ X.transpose()
        
    def train(self, X, y):
        mse_loss = 0
        self.m_samples, self.n_features = X.shape 
        X_aug = np.hstack((X, np.ones((self.m_samples, 1)))).transpose() #X = [X^T 1]^T of size 
        T = self.one_hot(y, self.num_classes).transpose() #one_hot encoded y_train
        self.W = np.zeros((self.num_classes, self.n_features +1)) # W = [w w0] of size C x (D + 1)
        not_converged = True
        while not_converged and self.num_iter < self.max_iter:
            self.num_iter += 1
            Z = self.W @ X_aug
            G = self.sigmoid(Z)
            grad_W = self.MSE_grad_W(G, T, X_aug)
            #Updating W-matrix
            self.W = self.W - self.alpha * grad_W 

            #logging MSE-loss per iteration
            prev_loss = mse_loss
            mse_loss = np.mean((G - T)**2) 

            #logging accuracy per iteration
            preds = np.argmax(G, axis=0)
            labels = np.argmax(T, axis=0) 
            accuracy = np.mean(preds == labels)

            #check convergence in mse
            if np.abs(prev_loss - mse_loss) < self.threshold: 
                not_converged = False
            self.losses.append(mse_loss)
            self.train_accuracies.append(accuracy)

                
    def test(self, X):
        m_test = X.shape[0] 
        X_aug = np.hstack((X, np.ones((m_test, 1)))).transpose()
        Z = self.W @ X_aug
        G = self.sigmoid(Z)
        return np.argmax(G, axis=0)
    
def find_error_rates(y_pred, y_labels):
    error_rate = np.sum(y_pred != y_labels) / len(y_labels)
    return error_rate

def confusion_matrix(y_labels, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(y_labels, y_pred):
        cm[true, pred] += 1
    
    return cm
    
#Functions for data preprocessing
def load_data():
    x1_df = pd.read_csv('class_1.csv')
    x2_df = pd.read_csv('class_2.csv')
    x3_df = pd.read_csv('class_3.csv')

    x1_df["class"] = 0
    x2_df["class"] = 1
    x3_df["class"] = 2

    full_df = pd.concat([x1_df, x2_df, x3_df], ignore_index=True)
    return full_df

def train_test_split(full_df, num_train, num_test):
    train_list, test_list = [], []

    for class_label in full_df['class'].unique():
        class_df = full_df[full_df['class'] == class_label]
        
        class_df = class_df.sort_index()
        
        if num_train >= 0:
            train_list.append(class_df.iloc[:num_train])
        else:
            train_list.append(class_df.iloc[num_train:])  

        if num_test >= 0:
            test_list.append(class_df.iloc[:num_test])
        else:
            test_list.append(class_df.iloc[num_test:])  

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, test_df

def feature_selection(train_df, test_df, exclude_features=None):  
    all_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    if exclude_features:
        if isinstance(exclude_features, str):
            exclude_features = [exclude_features]
        features = [f for f in all_features if f not in exclude_features]
    else:
        features = all_features

    X_train = train_df[features].values
    y_train = train_df['class'].values

    X_test = test_df[features].values
    y_test = test_df['class'].values

    return X_train, y_train, X_test, y_test
 
#Functions for plotting
def plot_loss(classifier):
    n_iter = np.arange(0, classifier.num_iter, 1)
    plt.plot(n_iter, classifier.losses)
    plt.xlabel("Iterations")
    plt.ylabel("MSE/accuracy")
    plt.title("Loss and accuracy")
    plt.grid()
    plt.show()

def plot_accuracies(clasifier):
    n_iter = np.arange(0, clasifier.num_iter, 1)
    plt.plot(n_iter, clasifier.train_accuracies)
    plt.xlabel("Iterations")
    plt.ylabel("MSE/accuracy")
    plt.title("Loss and accuracy")
    plt.grid()
    plt.show()

def plot_confusion_matrix(confusion_matrix, err_rate, dataset_name, num_iter):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Setosa', 'Versicolor','Virginica'], yticklabels=['Setosa', 'Versicolor','Virginica'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {dataset_name} set\nError rate: {err_rate:.3f} and {num_iter} iterations")    
    plt.show()


def plot_histograms(x_df):
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    class_labels = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    colors = ['blue', 'red', 'green']  # Nice readable colors

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Feature Histograms by Class', fontsize=18)

    for ax, feature in zip(axes.flatten(), features):
        for cls, color in zip(class_labels.keys(), colors):
            subset = x_df[x_df['class'] == cls]
            ax.hist(subset[feature], bins=10, alpha=0.5, label=class_labels[cls], color=color)
        
        ax.set_title(f'{feature}')
        ax.set_xlabel('Measurement [cm]')
        ax.set_ylabel('Number of samples')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_pairplot(x_df):
    sns.pairplot(x_df, hue = "class", palette = "deep")


def run_experiment(train_df, test_df, exclude_features=None, title='Task 1a'):
    X_train, y_train, X_test, y_test = feature_selection(train_df, test_df, exclude_features)
    
    classifier = LinearClassifier()
    classifier.train(X_train, y_train)

    y_train_pred = classifier.test(X_train)
    y_test_pred = classifier.test(X_test)

    err_rate_train = find_error_rates(y_train_pred, y_train)
    err_rate_test = find_error_rates(y_test_pred, y_test)
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title}\nTrain Error: {err_rate_train:.4f}, Test Error: {err_rate_test:.4f}', fontsize=18)

    n_iter = np.arange(0, classifier.num_iter, 1)

    # Plot loss
    axes[0, 0].plot(n_iter, classifier.losses)
    axes[0, 0].set_title('MSE-Loss over Iterations')
    axes[0, 0].set_xlabel('Iterations')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].grid(True)

    # Plot training accuracy
    axes[0, 1].plot(n_iter, classifier.train_accuracies)
    axes[0, 1].set_title('Training Accuracy over Iterations')
    axes[0, 1].set_xlabel('Iterations')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)

    # Train confusion matrix
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
                xticklabels=['Iris-setosa', 'Iris-versicolor','Iris-virginica'], yticklabels=['Iris-setosa', 'Iris-versicolor','Iris-virginica'])
    axes[1, 0].set_title(f"Confusion Matrix for Train-set\nError rate: {err_rate_train:.3f} and {classifier.num_iter} iterations")
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_ylabel('True Label')

    # Test confusion matrix
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1],
                xticklabels=['Iris-setosa', 'Iris-versicolor','Iris-virginica'], yticklabels=['Iris-setosa', 'Iris-versicolor','Iris-virginica'])
    axes[1, 1].set_title(f"Confusion Matrix for Test-set\nError rate: {err_rate_test:.3f}")
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_ylabel('True Label')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def task_1abc(x_df, num_train, num_test):
    train_df, test_df = train_test_split(x_df, num_train, num_test)
    run_experiment(train_df, test_df, title="Task 1abc")

def task_1d(x_df, num_train, num_test):
    train_df, test_df = train_test_split(x_df, num_train, num_test)
    run_experiment(train_df, test_df, title="Task 1d")

def task_2a(x_df):
    plot_histograms(x_df)
    num_train = 30
    num_test = -20
    train_df, test_df = train_test_split(x_df, num_train, num_test)
    run_experiment(train_df, test_df, exclude_features='sepal_width', title='Task 2a')
    #The feature with the most overlap is sepal width, training the classifier again without this feature 

def task_2b_1(x_df):
    #repaet with also removing sepal length 
    num_train = 30
    num_test = -20
    train_df, test_df = train_test_split(x_df, num_train, num_test)
    run_experiment(train_df, test_df, exclude_features=['sepal_width', 'sepal_length'], title='Task 2b_1')

def task_2b_2(x_df):
    #repaet with also removing petal width 
    num_train = 30
    num_test = -20
    train_df, test_df = train_test_split(x_df, num_train, num_test)
    run_experiment(train_df, test_df, exclude_features=['sepal_width', 'sepal_length', 'petal_length'], title='Task 2b_2')

def main():
    x_df= load_data()
    task_1abc(x_df, 30, -20)
    task_1d(x_df, -30, 20)
    task_2a(x_df)
    task_2b_1(x_df)
    task_2b_2(x_df)


if __name__ == '__main__':
    main()