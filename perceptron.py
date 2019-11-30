'''
Ram Paranjothy
Nov 28 2019
Rishi Neural Net
'''
import random
from data_gen import trainData as trainingData
from data_gen import valData as validationData
import math

class Perceptron(object):
    inputCt=None
    i=[]
    weight=[]
    bias=0
    lrate=0.1
    target=None

    def __init__(self,iCt,oCt):
        # Activate only once. 
        #  so for the second set of training data, the weights from first should be adjusted.
        self.inputCt=iCt
        self.activate()

       
    def activate(self):
        # randomly initialize weigths between 1 and -1
        self.weight=[random.uniform(-1,1) for _ in range(self.inputCt)]

    def guess(self):
        # compute the weighted sum for input and map to 1 and -1
        x=reduce(lambda x,o:o+x,map(lambda x:float(x[0])*float(x[1]), zip(self.weight,self.i)))
        # print x
        # xx=1/(1+math.exp(-x))
        # print xx
        # return x
        return 1 if x>=0 else -1

    def setData(self,x):
        #this will be used to set the input list and the target label to each observation
        trainInput=x[0]
        trainTgt=x[1]
        self.i=trainInput
        self.target=trainTgt

    def normalize(self):
        # maxInput=max([abs(_) for _ in self.i])
        # if maxInput>1 or abs(min(self.i)) > 1:
        #     self.i=[float(_)/float(maxInput) for _ in self.i]

        self.i=map(lambda x: 1/(1+math.exp(-x)),self.i)   

    def train(self,x):
        print("--"*10)
        self.setData(x)
        self.normalize()
        self._train()

    def _train(self):
        # This method mutates the weights if guess and target doesnt match
        print("\tCurrent Input - {0}".format(self.i))
        print("\tCurrent weights - {0}".format(self.weight))
        g=self.guess()
        error=float(g)-float(self.target)
        print ("\tError - {0} : Guess: {1} Target: {2}".format(error,g,self.target))
        # now tune weights per error -> weight = weight + error*each weight
        if float(error) != 0.0:  
            # increment if error is + and decrement if error is -ve
            self.weight=map(lambda x:(float(x)*float(error)*float(self.lrate))+float(x),self.i)  if error < 0 else map(lambda x:(float(x)*float(error)*float(self.lrate))-float(x),self.i)
            print("***** Tuning weights - {0}".format(self.weight))
            self._train()
        else:
            print("\tFinal weights : {0}".format(self.weight))
            return error
       
    def getWeights(self):
        return self.weight


    def predict(self,inputs,weight):
        '''
        THis is the predict guy
        '''
        # maxInput=max([abs(_) for _ in inputs[0]])
        lbl=inputs[1]
        # if maxInput>1 or abs(min(self.i)) > 1:
        #     inputsNormalized=[float(_)/float(maxInput) for _ in inputs[0]]

        inputsNormalized=map(lambda x: 1/(1+math.exp(-x)),inputs[0])   

        # for this to be accurate, your weight shuld have been trained.
        x=reduce(lambda x,o:o+x,map(lambda x:float(x[0])*float(x[1]), zip(weight,inputsNormalized)))
        out= 1 if x>=0 else -1
        # return ('Input: {0}\t| Label: {1}\t| Prediction: {2}\t| Matching: {3}\t'.format(inputs[0],lbl,out,lbl==out))
        return {
                'input':inputs[0],
                'label':lbl,
                'prediction':out,
                'matched':lbl==out
        }



if __name__ == "__main__":
    # trainingData=[([2,4,5],1),([2,4,-5],-1),([2,4,-44],-1),([2,4,55],1),([22,24,-55],-1),
    # ([12,14,5],1),([22,24,-5],-1),([32,34,-44],-1),([42,4,55],1),([22,24,-55],-1),([22,24,55],1)]
    p=Perceptron(2,1)    
    for d in trainingData:
        p.train(d)

    print("Training Done...")
    print

    tunedWeight=p.getWeights()
    print("Tuned Weight: {0}".format(tunedWeight))

    # validationData=[([20,40,50],1),([20,40,-50],-1),([200,400,-440],-1),([200,400,550],1),([202,204,-505],-1)]
    
    o=[p.predict(vData,tunedWeight) for vData in validationData]
    print(reduce(lambda x,o:o+'\n'+x,map(lambda x:'{0}|{1}|{2}|{3}'.format(x['matched'],x['input'],x['label'],x['prediction']),o)))
    # for vData in validationData:
    #     print(p.predict(vData,tunedWeight))




        
        