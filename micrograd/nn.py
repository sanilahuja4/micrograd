from micrograd.engine import Value
import random

class Neuron():
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1) for _ in range(nin))]
        self.b = Value(0)

    def __call__(self,x):
        

    



