import numpy as np
import optimizers

def square_plus_2(x):
    return - (x**2 + np.sin(x*np.pi))

optimizer = optimizers.DichotomicOptimizer(
    step=100,
    nb_recurse=5,
    nb_process=2,
)

optimizer.fit(square_plus_2, 
    kwds={
        "x": (-2, 2),
    },
)

best_y, best_score = optimizer.best
print("best_y: ", best_y)
print("calc: ", square_plus_2(**best_y))
print("min: ", np.min([square_plus_2(x) for x in np.linspace(-2, 2, 1000)]))
print("max: ", np.max([square_plus_2(x) for x in np.linspace(-2, 2, 1000)]))
