from utilz import *
from segment import *
from MAEC import MAEC
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data = get_data()
MAEC = MAEC()
MAEC.train(1000, 16, data)