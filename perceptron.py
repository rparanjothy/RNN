'''
Ram Paranjothy
Nov 28 2019
Rishi Neural Net
'''
import random

class Perceptron(object):
    i=[]
    o=[]
    weight=[]
    bias=0
    lrate=0.01

    def __init__(self,i):
        # initialize weights when a perceptron is created.
        #  so we can tune it as we train it
        if not isinstance(i,list): 
            raise Exception("Input should be a list")
        self.i=i
        self.activate()
        self.normalize()
            
        # self.output=output
  
    def activate(self):
        # randomly initialize weigths between 1 and -1
        self.weight=[format(random.uniform(-1,1),"1.2f") for _ in self.i]

    def guess(self):
        # compute the weighted sum for input and map to 1 and -1
        x=reduce(lambda x,o:o+x,map(lambda x:float(x[0])*float(x[1]), zip(self.weight,self.i)))
        return 1 if x>=0 else -1

    def normalize(self):
        maxInput=max([abs(_) for _ in self.i])
        if maxInput>1 or abs(min(self.i)) > 1:
            self.i=[format(float(_)/float(maxInput),'1.2f') for _ in self.i]
            print("*"*5)
            # print('Normalized inputs : {0}'.format(self.i))

    
    def train(self,target):
        # This method mutates the weights if guess and target doesnt match
        print("\tCurrent Input - {0}".format(self.i))
        print("\tCurrent weights - {0}".format(self.weight))
        g=self.guess()
        error=float(g)-float(target)
        print ("\tError - {0} : Guess: {1} Target: {2}".format(error,g,target))
        # now tune weights per error -> weight = weight + error*each weight
        if float(error) != 0.0:  
            # increment if error is + and decrement if error is -ve
            self.weight=map(lambda x:(float(x)*float(error)*float(self.lrate))+float(x),self.i)  if error < 0 else map(lambda x:(float(x)*float(error)*float(self.lrate))-float(x),self.i)
            print("***** Tuning weights - {0}".format(self.weight))
            self.train(target)
        else:
            print("\tFinal weights : {0}".format(self.weight))
            return error
       


if __name__ == "__main__":
    p=Perceptron([2,4,5])    
    p.train(1)