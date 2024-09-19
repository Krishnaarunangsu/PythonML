from scipy import  stats
from matplotlib import pyplot as plt

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err=stats.linregress(x,y)
print(f'Slope:{slope}')
print(f'Intercept:{intercept}')
print(f'R:{r}')
print(f'P-Value:{p}')
print(f'Standard Error:{std_err}')

def my_func(x_coord)->float:
    """

    Args:
        x_coord:

    Returns:

    """
    return slope*x_coord+intercept

my_model= list(map(my_func,x))

plt.scatter(x, y)
plt.plot(x,my_model)
plt.show()