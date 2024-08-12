"""Visualize the plots script."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.load_config import load_config
from src.utils.load_data import load_data
from io import BytesIO

def plot_dataset(X, y):
    """Plot the dataset with training and testing points."""
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99
    )
    h = 0.02 # meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    plt.figure(figsize=(10,8))
    plt.cm.PiYG
    cm_bright = ListedColormap(["#FF0000", "#00ff5e"])
    plt.title("Input data")
        
    # Plot the training points
    plt.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
    )
    # Plot the testing points
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, marker='x',  cmap=cm_bright, alpha=1 
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def plot_classifier_output(X, y, clf, score):
    """Plot the classifier output with decision boundaries."""
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 10))
    cm = plt.cm.PiYG
    cm_bright = ListedColormap(["#FF0000", "#00ff5e"])

    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_title("Classifier output")
    ax.text(xx.max() - 0.3, yy.min() + 0.3, (f"score = {score:.2f}").lstrip("0"), size=15, horizontalalignment="right")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def plot_inference_output(X_inference, y_pred, clf):
    """Plot the inference output with decision boundaries."""
    h = 0.02
    x_min, x_max = X_inference[:, 0].min() - 0.5, X_inference[:, 0].max() + 0.5
    y_min, y_max = X_inference[:, 1].min() - 0.5, X_inference[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 10))
    cm = plt.cm.PiYG
    cm_bright = ListedColormap(["#FF0000", "#00ff5e"])

    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    ax.scatter(X_inference[:, 0], X_inference[:, 1], marker="x", c=y_pred, cmap=cm_bright)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_title("Classifier inference output")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf
