# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import i_utils
import operator
"""T可以理解为高维数组，里面其实有scalar（标量），vector (向量），matrix (矩阵），tensor3 (三维矩阵)，tensor4（四位矩阵）"""

class RNNTheano:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        #Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        #Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        #Theano:Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        #We store the Theanp graph here
        self.theano = {}
        self.__theano_build__()
        
    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev)) #取出x_t这一列的U
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
            #return [o_t, s_t]
            #print o_t[0]
        #需要注意的是scan在计算的时候，可以访问以前n步的输出结果，所以比较适合RNN网络。
        #theano.scan(fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, truncate_gradient=-1, strict=False)
        #fn:是一个lmbda或者def函数
        #sequence:是一个theano variable或者dictionaries的列表
        #outputs_info:是一个theano variables或者dictionaries的列表，它描述了输出的初始状态，每进行一步scan
        #操作，outputs_info中的数值会被上一次迭代的输出装值更新掉。
        #non_sequence是一个常量参数列表，代表了一步scan操作中不会被更新的变量。
        #truncate_gradient:参数代表了使用BPTT（backpropagation through time)算法时，“梯度截断”后的步数。
        #“梯度截断”的目的是在可接受的误差范围内， 降低梯度的计算复杂度。
        #strict:是一个shared variable校验标志，用于检验是否fn函数用到的所有shared variables都在non_sequences中
        [o,s], updates = theano.scan(
                forward_prop_step,
                sequences=x,
                outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
                non_sequences=[U, V, W],
                truncate_gradient=self.bptt_truncate,
                strict=True)
        
        #argmax(o, axis=1)表示o[i][0],o[i][1],o[i][2],o[i][3],...
        prediction = T.argmax(o, axis=1) #将输出概率最大的设为预测值
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        #Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        
        #Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])
        
        #SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x, y, learning_rate], [],
                                        updates=[(self.U, self.U - learning_rate * dU),
                                                 (self.V, self.V - learning_rate * dV),
                                                 (self.W, self.W - learning_rate * dW)])
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])
    def calculate_loss(self, X, Y):
        #Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y)/float(num_words)
    

def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    #Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    """每当进行反向传播时，最好也进行梯度检查，这是一种验证你的实现是否正确的方法。梯度检查背后的想法是参数的导数
    等于该点的斜率，我们可以通过稍微改变参数来近似： 
    ∂L/∂θ≈limh→0[J(θ+h)−J(θ−h)]/2h
    然后比较使用反向传播计算的梯度和用上述方法估计的梯度。如果没有大的差别，那我们的计算是正确的。
    近似计算需要计算每个参数的总损耗，因此梯度检查代价非常昂贵（在上面的例子中我们有超过一百万个参数）。
    所以在一个较小的词汇模型上执行梯度检查是一个好主意。"""
    model.bptt_truncate = 1000
    #Calculate the gradient using backprop
    bptt_gradients = model.bptt(x, y)
    #List of all parameters we want to check
    model_parameters = ['U', 'V', 'W']
    #Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        #Get the actual parameter value from the model, eg, model.w
        #f = op.attrgetter(‘name’)   f(person)  #return person.name
        #f = op.attrgetter(‘name’,’age’)    f(person)  #return a tuple of (person.name,person.age)
        parameter_T = operator.attrgetter(pname)(model) #f=operator.attrgetter(pname) f(model)  return model.pname
        parameter = parameter_T.get_value()
        print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        #Iterate over each element of the parameter matrix, eg,(0,0),(0,1),...
        #np.nditer：默认行序优先迭代遍历
        it = np.nditer(parameter, flag=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            #Save the original value so we can reset it later
            original_value = parameter[ix]
            #Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            #The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            #Calculate the relative error:(|x - y|)/(|x| + |y|)
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            #If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                print "+h Loss: %f" % gradplus
                print "-h Loss: %f" %gradminus
                print "Estimated_gradient: %f" % estimated_gradient
                print " Backpropagation gradient: %f" % backprop_gradient
                print "Relative Error: %f" % relative_error
                return
            it.iternext()
        print "Gradient check for parameter %s passed." % (pname)
            
    
        
        