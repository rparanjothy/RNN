import numpy as np
class RNN(object):
    weight1=None
    weight2=None
    weight=[weight1,weight2]
    X=None
    H=None
    Y=None

    def __init__(self,inputCt,hiddenCt,outputCt):
        self.inputCt=inputCt
        self.outputCt=outputCt
        self.hiddenCt=hiddenCt
        self.X=np.zeros(inputCt)
        self.H=np.zeros(hiddenCt)
        self.Y=np.zeros(outputCt)

    def initializeWeights(self):
        print("Initializing weights")
        self.weight1=np.random.random([self.hiddenCt,self.inputCt])
        self.weight2=np.random.random([self.outputCt,self.hiddenCt])
        self.weight1=self.weight1*2-1
        self.weight2=self.weight2*2-1
        self.b1=np.random.random([self.hiddenCt,1])
        self.b2=np.random.random([self.outputCt,1])

        print 'W1\n',self.weight1
        print 'W2\n',self.weight2


    def train(self,x):
        print("Training")
        self.initializeWeights()
        self.X=np.array(x[0]).reshape(self.inputCt,1)
        self.H=np.dot(self.weight1,self.X)
        # print 'B1:\n',self.b1
        # self.H=np.add(self.H,self.b1)
        #print 'H:\n',self.H
        self.H=(1/(1+np.exp(-self.H)))
        #print 'H (Sigmoid):\n',self.H

        self.Y=np.array(x[1]).reshape(self.outputCt,1)
        guess=np.dot(self.weight2,self.H)
        # guess=np.add(guess,self.b2)
        # #print 'B2:\n',self.b2
        #print 'Computed\n',guess
        g=(1/(1+np.exp(-guess)))
        #print 'sigmoid:\n',g
        err=self.Y-g
        #print 'Err\n',err

        w2t=np.transpose(self.weight2)
        hiddenError=np.dot(w2t,err)
        #print 'HiddenError\n',hiddenError
        #print "Gradients\n"
        
        sigG=(1/(1+np.exp(g)))
        guessDerivative=sigG*(1-sigG)
        #print guessDerivative

       
        i=guessDerivative*0.1*err  
        #print 'GuessDer*lr*error\n',i


        sigH=(1/(1+np.exp(self.H)))
        HDerivative=sigH*(1-sigH)
        #print 'HDerivative\n',HDerivative

        o1=np.dot(np.transpose(g),i)
        #print 'o1\n',o1


        
        j=HDerivative*0.1*hiddenError  
        #print 'HiddenDer*lr*error\n',j

        h1=np.dot(np.transpose(self.H),j)
        #print 'h1\n',h1

        self.weight1=self.weight1+h1
        self.weight2=self.weight2+o1

        print 'after backpro'
        print "-="*40
        print 'ComputedSigmoided\n',g
        print 'Err\n',err
        

        print 'w1:\n',self.weight1
        print 'w1:\n',self.weight2



    def validate(self,x):
        print("Validating..")
    
    def predict(self,x):
        print("Predicting...")

    @staticmethod
    def fromWeights(ict,hct,oct,w):
        x=RNN(ict,hct,oct)
        x.weight=w
        return x
            
if __name__ == "__main__":
    rnn=RNN(2,4,2)
    
    trainingData=[([1,1],[1,0]),([0,1],[1,0]),([1,0],[0,0]),([1,1],[1,0]),([0,1],[1,0]),([1,0],[0,0]),([1,1],[1,0]),([0,1],[1,0]),([1,0],[0,0])]
    for x in trainingData:
        rnn.train(x)
    # rnn.validate(validationData)
    # rnn.predict(predictInputs)
