<h1>Recap Jensen's Ineqaulity - A statistical simulation of convex functions for an intuitive understanding</h1>
  
Jensen's Ineqaulity is an equation that holds for every convex function (and value range of non-strictly convex functions). It is a key point to understand the derivation of the <i>Evidence Lower Bound (ELBO)</i> in Variational Inference of Blei et al. (2016).
Jensen's Inequality states the fact that the function outcome of a convex function with the expected value of a random variable <i>X</i> as input is always equal or smaller than the expected value of the convex function outcome.

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{110}&space;f(E(X))&space;\leq&space;E(f(X))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;f(E(X))&space;\leq&space;E(f(X))" title="f(E(X)) \leq E(f(X))" /></a>

This statement is very abstract and it is not trivial to understand. Therefore, let's look at an example. We assume that the random variable <i>X</i> is randomly drawn from a normal distribution with &mu; = 3 and &sigma; = 1. In our example we draw n = 100 samples.

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{110}&space;X&space;\sim&space;N(3,1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;X&space;\sim&space;N(3,1)" title="X \sim N(3,1)" /></a>

```python
import numpy as np
import pylab
```

```python
mean = 3
std = 1
n = 100
```

Next step is to define a function and its reverse function. In our example we choose the following convex function:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{110}&space;f(X)&space;=&space;x^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;f(X)&space;=&space;x^2" title="f(X) = x^2" /></a>

```python
def f(x):
    return x**2

def f_reverse(y):
    return y**(1/2)
```

Let's draw the function for the positive and negative value range:

```python
x_negative = np.linspace(-15,-0.0,100) # 100 linearly spaced numbers
x_positive = np.linspace(0.0,15,100) # 100 linearly spaced numbers
y_negative = f(x_negative)
y_positive = f(x_positive)

pylab.plot(x_negative,y_negative, color="C0")
pylab.plot(x_positive,y_positive, color="C0")
pylab.show() # show the plot
```
<img src="img_x2.png"></img>

Now we define a function to sample random values from our normal distribution:

```python
def gen_data(mean, std, n):
    x = np.random.normal(mean,std,(n)) # sample 100 values
    y = []

    e_x = np.mean(x) # compute E(x)
    for _x in x:
        y.append(f(_x))
    e_f_x = np.mean(y)  # compute E(f(x))
    return x, y, e_x, e_f_x
 ```
 
Next step is to sample positive values for <i>x</i> and <i>y</i> and draw a plot with f(E(x)) and E(f(x)). For example, if <i>x</i> = [1,2] these values are computed as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;E(X)&space;=&space;\frac{1&plus;2}{2}=1.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;E(X)&space;=&space;\frac{1&plus;2}{2}=1.5" title="E(X) = \frac{1+2}{2}=1.5" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;f(E(X))&space;=&space;f(1.5)&space;=&space;1.5^2&space;=&space;2.25" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;f(E(X))&space;=&space;f(1.5)&space;=&space;1.5^2&space;=&space;2.25" title="f(E(X)) = f(1.5) = 1.5^2 = 2.25" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;f([1,2])&space;=&space;[1,4]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;f([1,2])&space;=&space;[1,4]" title="f([1,2]) = [1,4]" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;&space;E(f(x))&space;=&space;\frac{1&plus;4}{2}&space;=&space;2.5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;&space;E(f(x))&space;=&space;\frac{1&plus;4}{2}&space;=&space;2.5" title="{\color{Red} E(f(x))} = \frac{1+4}{2} = 2.5" /></a>

 
```python
x, y, e_x, e_f_x = gen_data(mean,std,n) # sample data for positive value range
pylab.scatter(x, y, s=2)
f_e_x_label = pylab.scatter(e_x, f(e_x), s=50, color='black', label='f(E(x)) = ' + str(round(f(e_x),2)))
e_f_x_label = pylab.scatter(f_reverse(e_f_x), e_f_x, s=50, color='red', label='E(f(x)) = ' + str(round(e_f_x,2)))
pylab.legend(handles=[f_e_x_label, e_f_x_label])
pylab.show() # show the plot
```
<img src="img_x2_positive.png"></img>

As you can see in the plot, the red point is above the black point. Therefore, the condition of Jensen's Inequality holds, no matter how often the experiment is repeated:

```python
f(e_x) <= e_f_x
```
```
True
```

We can also plot it for negative values:
<img src="img_x2_negative.png"></img>

```python
f(e_x) <= e_f_x
```
```
True
```

The function in the above plots is convex for positive and negative values. But there are functions in which the convexity is dependend on its value range. Let's plot different functions:

<img src="img_functions.png"></img>

We can see that Jensens's Inequality also holds for different value ranges of non-strictly convex functions. I hope you enjoyed reading this article and got a good understanding what is meant by Jensen's Inequality.
