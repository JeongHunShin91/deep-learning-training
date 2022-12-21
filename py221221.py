# 노드
import numpy as np
D, N = 8, 7
x = np.random.randn(1, D)
x
y = np.repeat(x,N, axis=0)
y
dy = np.random.randn(N, D)
dy
dx = np.sum(dy, axis = 0, keepdims = True)
dx

D, N = 8, 7
x = np.random.randn(N, D)
y = np.sum(x, axis = 0, keepdims = True)
x
y
dy = np.random.randn(1, D)
dx = np.repeat(dy, N, axis = 0)
dy
dx

class matmul :
    def __int__(self,w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.x = None

    def forward(self, x):
        w, = self.params
        out = np.matmul(x, w)
        self.x = x
        return out

    def backward(self, dout):
        w, = self.params
        dx = np.matmul(dout, w.t)
        dw = np.matmul(self.x.t, dout)
        self.grads[0][...] = dw
        return dx

class sigmoid :
    def __int__(self):
        self.params, self.grads = [],[]
        self.out = None

    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out

    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class affine :
    def __int__(self,w,b):
        self.params =[w,b]
        self.grads = [np.zeros_like(w), np.zeros_like(b)]
        self.x = None

    def forward(self,x):
        w, b = self.params
        out = np.matmul(x,w) + b
        self.x = x
        return out

    def backward(self,dout):
        w, b = self.params
        dx = np.matmul(dout, w.t)
        dw = np.matmul(self.x.t, dout)
        db = np.sum(dout, axis = 0)

        self.grads[0][...] = dw
        self.grads[1][...] = db
        return dx

class SGD :
    def ___init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i]-= self.lr * grads[i]

model = twolayernet(...)
optimizer = SGD()

for i in range(1000):
    ...
    x_batch, t_batch = get_mini_batch(...)
    loss = model.forward(x_batch, t_batch)
    model.backward()
    optimizer.update(model.params,model.grads)
    ...
