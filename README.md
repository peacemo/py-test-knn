# Python KNN: 从头开始创建一个 Python 包

## 项目初始化

- `poetry init` 初始化项目结构

- `poetry source add --default mirrors https://mirrors.bfsu.edu.cn/pypi/web/simple/` 添加国内源

- `poetry add [lib name]` 添加依赖

- `poetry add [lib name] --group test` 添加测试环境依赖

- `poetry run python` 运行当前项目环境下的 python

- `poetry run pytest` 运行单元测试

Python 包需要:

- 同名文件夹 `ty_kmeans`: `packages = [{include = "ty_kmeans"}]`
- 文件夹下必须要有 `__init__.py` (可以为空)
  - `__init__.py` 中可以有 `__all__ = [...]` 来指定用户可以获取的符号（变量、函数、模块）

## 安装本项目

- 安装：`pip install git+https://github.com/peacemo/py-test-knn.git`

- 使用

```julia
from tyknn import knn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

K = 5
y_pred = knn(X_train, y_train, X_test, K)

confusion_matrix(y_test, y_pred)

>>> output
array([[19,  0,  0],
       [ 0, 15,  0],
       [ 0,  1, 15]])

```