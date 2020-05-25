import random
import numpy as np



if __name__ == '__main__':
    experiences = []
    for i in range(32):
        # a = random.randint(0, 100)
        b = np.random.uniform(0, 1, 1)
        # c = random.randint(0, 100)
        experiences.append(b)

    s = np.vstack(experiences)
    pass