from som_class import *
T1=np.random.rand(1,1000)
T2=np.random.rand(1,1000)

P=
X=np.concatenate((np.cos(2*np.pi*T),np.sin(2*np.pi*T)),axis=0)
W=np.random.rand(2,20)

som=SOM(W)
som.set_lattice("circle",20)
som.train(X,200)
som.make_plot(W,X)

som.set_lattice("rectangle",5,4)
som.train(X,200)
som.make_plot(W,X)

# som.set_lattice("circle",20)
# som.train(X,10)
# som.plotoutput(W,X)



plt.show()