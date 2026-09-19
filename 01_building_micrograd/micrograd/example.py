from nn import MLP

learning_rate = 0.05

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
n = MLP(3, [4,4,1])


for k in range(1, 101):
    # forward prop
    ypred = [n(x) for x in xs] 
    loss = sum(((ygt-yout[0])**2 for ygt, yout in zip(ys, ypred))) / len(ys)

    for p in n.parameters():
        p.grad = 0 # important!!!!

    # back prop
    loss.backward()

    # update
    for p in n.parameters():
        p.data -= learning_rate * p.grad
        p.grad = 0 # important!!!!
    if k % 10 == 0:
        print(f'k={k}, loss={loss.data}')

print(n.parameters())