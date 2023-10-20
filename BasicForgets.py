import sys
sys.path.append('../')
import numpy as np
from pyBKT.models import Model
import json
from icecream import ic

np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    model = Model(seed = 0, num_fits = 20)
    model.fit(data_path = "data/ct.csv")
    print("Standard BKT:", model.evaluate(data_path = "data/ct.csv", metric="accuracy"))
    model2 = Model(seed = 0, num_fits = 20)
    model2.fit(data_path = "data/ct.csv", forgets=True)
    print("BKT+Forgets:", model2.evaluate(data_path = "data/ct.csv", metric="accuracy"))

    print("Without forgets:")
    ic(model.coef_)

    print("\nWith forgets:")
    ic(model2.coef_)

