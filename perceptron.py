import random
import math

class Perceptron(object):
    i=[]
    o=[]
    weight=[]
    bias=0
    lrate=0.1

    def __init__(self,i):
        # initialize weights when a perceptron is created.
        #  so we can tune it as we train it
        if not isinstance(i,list): 
            raise Exception("Input should be a list")
        self.i=i
        self.activate()
        self.normalize()
            
        # self.output=output
    def show(self):
        print 'Input {0} - Output {1}'.format(self.i,self.o)    
    
    def activate(self):
        # randomly initialize weigths between 1 and -1
        self.weight=[format(random.uniform(-1,1),"1.4f") for _ in self.i]

    def guess(self):
        # compute the weighted sum for input and map to 1 and -1
        x=reduce(lambda x,o:o+x,map(lambda x:float(x[0])*float(x[1]), zip(self.weight,self.i)))
        return 1 if x>=0 else -1

    def normalize(self):
        maxInput=max([abs(_) for _ in self.i])
        if maxInput>1 or abs(min(self.i)) > 1:
            self.i=[format(float(_)/float(maxInput),'1.5f') for _ in self.i]
            print("*"*5)
            # print('Normalized inputs : {0}'.format(self.i))

    
    def train(self,target):
        # This method mutates the weights if guess and target doesnt match
        print("\tCurrent Input - {0}".format(self.i))
        # print("\tCurrent weights - {0}".format(self.weight))
        g=self.guess()
        error=float(g)-float(target)
        print ("\tError - {0} : Guess: {1} Target: {2}".format(error,g,target))
        # now tune weights per error -> weight = weight + error*each weight
        if float(error) == 0.0:
            print("\tFinal weights : {0}".format(self.weight))
            return error
        else:
            self.weight=map(lambda x:(float(x)*float(error)*float(self.lrate))+float(x),self.i)
            print("***** Tuning weights - {0}".format(self.weight))
            return self.train(target)


if __name__ == "__main__":
    p=Perceptron([.5,4.10])    
    p.train(1)