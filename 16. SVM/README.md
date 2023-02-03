The **Support Vector Machine** is a **supervised learning algorithm** mostly used for **classification** but it can be used also for **regression.** The main idea is that based on the labeled data (training data) the algorithm tries to find the **optimal hyperplane** which can be used to classify new data points. In two dimensions the hyperplane is a simple line.

Usually a learning algorithm tries to learn the **most common characteristics (what differentiates one class from another)** of a class and the classification is based on those representative characteristics learnt (so classification is based on differences between classes). The **SVM** works in the other way around. It finds the **most similar examples** between classes. Those will be the **support vectors.**

As an example, lets consider two classes, apples and lemons.

Other algorithms will learn the most evident, most representative characteristics of apples and lemons, like apples are green and rounded while lemons are yellow and have elliptic form.

In contrast, SVM will search for apples that are very similar to lemons, for example apples which are yellow and have elliptic form. This will be a support vector. The other support vector will be a lemon similar to an apple (green and rounded). So **other algorithms** learns the **differences** while SVM learns **similarities.**

![image](https://user-images.githubusercontent.com/87495134/216513528-39fc4de7-afa9-4305-832c-a1df51861262.png)

As we go from left to right, all the examples will be classified as apples until we reach the yellow apple. From this point, the confidence that a new example is an apple drops while the lemon class confidence increases. When the lemon class confidence becomes greater than the apple class confidence, the new examples will be classified as lemons (somewhere between the yellow apple and the green lemon).

Based on these support vectors, the algorithm tries to find the **best hyperplane that separates the classes.** In 2D the hyperplane is a line, so it would look like this:

![image](https://user-images.githubusercontent.com/87495134/216513607-0ef9ed95-97f8-4a39-8c2b-9b3a118f4793.png)

## Finding the Optimal Hyperplane

Intuitively the **best line** is the line that is **far away from both apple and lemon examples** (has the largest margin). To have optimal solution, we have to **maximize the margin in both ways** (if we have multiple classes, then we have to maximize it considering each of the classes).

![image](https://user-images.githubusercontent.com/87495134/216513738-8de7a472-e7f9-490b-ba68-203ec8a725ec.png)

This margin is **orthogonal** to the boundary and **equidistant** to the support vectors.

## Support Vectors

**Support vectors** are data points that **defines the position and the margin of the hyperplane.** We call them **“support” vectors,** because these are the representative data points of the classes, **if we move one of them, the position and/or the margin will change.** Moving other data points won’t have effect over the margin or the position of the hyperplane.

So basically the **learning is equivalent with finding the hyperplane with the best margin**, so it is a **simple optimization problem.**

## Basic Steps
The basic steps of the SVM are:

= select **two hyperplanes** (in 2D) which separates the data **with no points between them** (red lines)
- **maximize their distance** (the margin)
- the **average line** (here the line half way between the two red lines) will be the **decision boundary**

To solve the optimization problem, we use the **Lagrange Multipliers.**

## SVM for Non-Linear Data Sets
An example of non-linear data is:

![image](https://user-images.githubusercontent.com/87495134/216514453-a9cba5da-5110-411b-8c0a-80318097f6a5.png)

In this case we **cannot find a straight line** to separate apples from lemons. So how can we solve this problem. We will use the **Kernel Trick!**

The basic idea is that when a data set is inseparable in the current dimensions, **add another dimension**, maybe that way the data will be separable. Just think about it, the example above is in 2D and it is inseparable, but maybe in 3D there is a gap between the apples and the lemons, maybe there is a level difference, so lemons are on level one and apples are on level two. In this case, we can easily draw a separating hyperplane (in 3D a hyperplane is a plane) between level 1 and 2.

## Mapping to Higher Dimensions
To solve this problem we **shouldn’t just blindly add another dimension**, we should transform the space so we generate this level difference intentionally.

## Mapping from 2D to 3D
Let's assume that we add another dimension called **X3**. Another important transformation is that in the new dimension the points are organized using this formula **x1² + x2²**.

If we plot the plane defined by the x² + y² formula, we will get something like this:
![image](https://user-images.githubusercontent.com/87495134/216514639-b817525f-05bc-4dea-ba85-edc8955c648c.png)


Now we have to map the apples and lemons (which are just simple points) to this new space. Think about it carefully, what did we do? We just used a transformation in which **we added levels based on distance**. If you are in the origin, then the points will be on the lowest level. As we move away from the origin, it means that we are **climbing the hill** (moving from the center of the plane towards the margins) so the level of the points will be higher. Now if we consider that the origin is the lemon from the center, we will have something like this:
![image](https://user-images.githubusercontent.com/87495134/216514709-bad74755-4d8e-40bb-81bb-5201b6143a01.png)

Now we can easily separate the two classes. These transformations are called **kernels**. Popular kernels are: **Polynomial Kernel, Gaussian Kernel, Radial Basis Function (RBF), Laplace RBF Kernel, Sigmoid Kernel, Anove RBF Kernel**, etc (see Kernel Functions or a more detailed description Machine Learning Kernels).

## Mapping from 1D to 2D
Another, easier example in 2D would be:
![image](https://user-images.githubusercontent.com/87495134/216514767-23222d1e-2d93-45f8-865b-f1ebebaa9359.png)

After using the kernel and after all the transformations we will get:
![image](https://user-images.githubusercontent.com/87495134/216514787-a9ec7e95-6158-4b51-86fb-9f4e3353a6d9.png)

So after the transformation, we can easily delimit the two classes using just a single line.

In real life applications we won’t have a simple straight line, but we will have lots of curves and high dimensions. In some cases we won’t have two hyperplanes which separates the data with no points between them, so we need some **trade-offs, tolerance for outliers**. Fortunately the SVM algorithm has a so-called **regularization parameter** to configure the trade-off and to tolerate outliers.

## Tuning Parameters
As we saw in the previous section **choosing the right kernel is crucial**, because if the transformation is incorrect, then the model can have very poor results. As a rule of thumb, **always check if you have linear data** and in that case always use **linear SVM** (linear kernel). **Linear SVM is a parametric model**, but an **RBF kernel SVM isn’t**, so the complexity of the latter grows with the size of the training set. Not only is **more expensive to train an RBF kernel SVM**, but you also have to keep the **kernel matrix around**, and the **projection into this “infinite” higher dimensional space** where the data becomes linearly separable is **more expensive** as well during prediction. Furthermore, you have **more hyperparameters to tune**, so model selection is more expensive as well! And finally, it’s much **easier to overfit a complex model!**

## Regularization
The **Regularization Parameter (in python it’s called C)** tells the SVM optimization **how much you want to avoid misclassifying** each training example.

If the **C is highe**r, the optimization will choose **smaller margin** hyperplane, so training data **misclassification rate will be lower**.

On the other hand, if the **C is low**, then the **margin will be big**, even if there **will be misclassified** training data examples. This is shown in the following two diagrams:
![image](https://user-images.githubusercontent.com/87495134/216515200-1211a7c4-1451-4cbf-a17a-fea1500fdd36.png) ![image](https://user-images.githubusercontent.com/87495134/216515205-984dce84-6f97-4700-9c6d-3f92eb0b3297.png)



As you can see in the image, when the C is low, the margin is higher (so implicitly we don’t have so many curves, the line doesn’t strictly follows the data points) even if two apples were classified as lemons. When the C is high, the boundary is full of curves and all the training data was classified correctly. **Don’t forget**, even if all the training data was correctly classified, this doesn’t mean that increasing the C will always increase the precision (because of overfitting).

## Gamma
The next important parameter is **Gamma**. The gamma parameter defines **how far the influence of a single training example reaches**. This means that **high Gamma** will consider only **points close to the plausible hyperplane** and **low Gamma** will consider **points at greater distance.**
![image](https://user-images.githubusercontent.com/87495134/216515462-edce4dfa-cf12-4ee9-8f50-2c3cfdb70675.png) ![image](https://user-images.githubusercontent.com/87495134/216515476-60e017b4-7bc2-441a-9209-b4f10c15e80e.png)

As you can see, decreasing the Gamma will result that finding the correct hyperplane will consider points at greater distances so more and more points will be used (green lines indicates which points were considered when finding the optimal hyperplane).

## Margin
**Higher margin results better model, so better classification (or prediction). The margin should be always maximized.**

## SVM Example using Python
Because the **sklearn** library is a very well written and useful Python library, we don’t have too much code to change. The only difference is that we have to import the **SVC** class (SVC = SVM in sklearn) from **sklearn.svm** instead of the KNeighborsClassifier class from sklearn.neighbors.

```
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 0.1, gamma = 0.1)
classifier.fit(X_train, y_train)
```

After importing the SVC, we can create our new model using the predefined constructor. This constructor has many parameters, but I will describe only the most important ones, most of the time you won’t use other parameters.

The most important parameters are:

- **kernel:** the kernel type to be used. The most common kernels are **rbf** (this is the default value), **poly** or **sigmoid**, but you can also create your own kernel.
- **C**: this is the **regularization parameter** described in the Tuning Parameters section
- **gamma**: this was also described in the Tuning Parameters section
- **degree**: it is used **only if the chosen kernel is poly** and sets the degree of the polinom
- **probability**: this is a boolean parameter and if it’s true, then the model will return for each prediction, the vector of probabilities of belonging to each class of the response variable. So basically it will give you the **confidences for each prediction**.
- **shrinking**: this shows whether or not you want a **shrinking heuristic** used in your optimization of the SVM, which is used in Sequential Minimal Optimization. It’s default value is true, an **if you don’t have a good reason, please don’t change this value to false**, because shrinking will **greatly improve your performance**, for very **little loss** in terms of **accuracy** in most cases.

Now lets see the output of running this code. The decision boundary for the training set looks like this:
![image](https://user-images.githubusercontent.com/87495134/216516005-5a788480-b72f-42bb-a1d3-f10efaedd015.png)

As we can see and as we’ve learnt in the Tuning Parameters section, because the C has a small value (0.1) the decision boundary is smooth.

Now if we increase the C from 0.1 to 100 we will have more curves in the decision boundary:
![image](https://user-images.githubusercontent.com/87495134/216516025-4cc0d6da-ca2d-45ff-84c2-6a1cd79069b7.png)

What would happen if we use C=0.1 but now we increase Gamma from 0.1 to 10? Lets see!
![image](https://user-images.githubusercontent.com/87495134/216516043-8edb144c-39db-4286-a2e6-130fa7cc1945.png)

What happened here? Why do we have such a bad model? As you’ve seen in the Tuning Parameters section, **high gamma** means that when calculating the plausible hyperplane we consider **only points which are close**. Now because the **density** of the green points is **high only in the selected green region,** in that region the points are close enough to the plausible hyperplane, so those hyperplanes were chosen. Be careful with the gamma parameter, because this can have a very bad influence over the results of your model if you set it to a very high value (what is a “very high value” depends on the density of the data points).

For this example the best values for C and Gamma are 1.0 and 1.0. Now if we run our model on the test set we will get the following diagram:
![image](https://user-images.githubusercontent.com/87495134/216516140-b29f7359-854d-49bd-a728-67c4513367c6.png)

And the Confusion Matrix looks like this:
![image](https://user-images.githubusercontent.com/87495134/216516152-8b34ead8-35ab-4c9a-b24e-87ae5d88c57d.png)

As you can see, we’ve got only **3 False Positives** and only **4 False Negatives**. The **Accuracy** of this model is **93%** which is a really good result, we obtained a better score than using KNN (which had an accuracy of 80%).

**NOTE:** accuracy is not the only metric used in ML and also **not the best metric to evaluate a model**, because of the Accuracy Paradox. We use this metric for simplicity!

## Pros
- SVN can be very **efficient**, because it uses only a subset of the training data, only the support vectors
- Works very well on **smaller data sets, on non-linear data sets** and **high dimensional spaces**
- Is very **effective** in cases where **number of dimensions is greater than the number of samples**
- It can have **high accuracy**, sometimes can perform even better than neural networks
- Not very **sensitive to overfitting**

## Cons
- **Training time is high** when we have large data sets
- When the data set has more **noise** (i.e. target classes are overlapping) **SVM doesn’t perform well**

## Popular Use Cases
Text Classification
Detecting spam
Sentiment analysis
Aspect-based recognition
Aspect-based recognition
Handwritten digit recognition

[Original Article](https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200)
