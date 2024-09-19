from scipy import  stats
from matplotlib import pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

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