import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import copy

imgpath='./Images/'


#Define the learning rate epsilon
def eps(n=1,T=None):
	return .9*(1-n/T)

#Define the parameter sigma - the width of Gaussian 
def sig(n=None,T=None):
	return .9*(1-n/(T+1))

#Function to pick winning center.  
def winning_center( x, W ):
	dist=float("inf")
	win_ind=-1
	for i, col in enumerate(W.T):
		dist_curr=la.norm(col-x)
		if dist_curr < dist:
			dist=dist_curr
			win_ind=i
	return win_ind


#Returns the coordinates in Euclidean space of a given index on a lattice
def lat_coords_line(lat,w_ind):
	return w_ind

def lat_coords_rectangle(lat,w_ind):
	return w_ind%lat[1], np.floor(w_ind/lat[1])

def lat_coords_circle(lat,w_ind):
	return w_ind

def lat_coords_torus(lat,w_ind):
	x=lat[1]
	y=lat[2]
	if w_ind==0:
		return np.array([[0,0,x-1,x-1],[0,y-1,0,y-1]])
	elif w_ind<x-1:
		return np.array([[w_ind,w_ind],[0,y-1]])
	elif w_ind%(x-1)==0:
		return np.array([[0,x-1],[w_ind%(x-1),w_ind%(x-1)]])
	else:
		return np.array([[w_ind%(x-1)],[np.floor(w_ind/lat[1])]])

#The following functions take two lattice indices x and y (not coordinates) and returns the
#lattice distance between those two indexes

def lat_dist_line(lat,x,y):
	return abs(x-y)

def lat_dist_rectangle(lat,x,y):
	return la.norm(np.array(lat_coords_rectangle(lat,x))-np.array(lat_coords_rectangle(lat,y)),2)

def lat_dist_circle(lat,x,y):
	if np.abs(x-y)==lat[1]-1:
		return 1
	else:
		return np.abs(x-y)

def lat_dist_torus(lat,x,y):
	z=lat_coords_torus(lat,x)
	w=lat_coords_torus(lat,y)
	min_dst=float("inf")
	for a in z.T:
		for b in w.T:
			dst=la.norm(a-b)
			if dst < min_dst:
				min_dst=dst
	return min_dst

#Function to update all centers given an input pattern x
def update_centers( W, win_ind, x, iter, lat,t,lattice_distance):
	for i, col in enumerate(W.T):
		w_new=eps(iter,t)*np.exp(-1*lattice_distance(lat,win_ind,i)**2/sig(iter,t)**2)*(x-col)+col
		W[:,i]=w_new
	return



class SOM(object):
	'Create a self organizing map object'

	def __init__(self,W=None):
		self.weights=copy.copy(W)

		if self.weights is not None:
			self.num_centers=self.weights.shape[1]



	def set_lattice(self,*params):
		"""line params: number of points to use
	       rectangle params: x and y dimensions of lattice
	       circle params: number of points to use
	       torus params: x and y dimensions of flat torus grid
		"""
		lat_nm=params[0]

		valid_nms=["line","rectangle","circle","torus"]
		if lat_nm not in valid_nms:
			print("lattice name must be one of: ", valid_nms)
			quit()

		self.lattice_nm=params[0]
		self.lattice=list(params)

		#if params[0]=="torus":








	#Train SOM on training data X
	def train(self,X,num_iters):
		#The lattice_distance function (called when updating centers) needs to be set depending on lattice
		lattice_dct={"line": lat_dist_line, "rectangle": lat_dist_rectangle, "circle": lat_dist_circle, "torus": lat_dist_torus}
		lattice_distance=lattice_dct[self.lattice[0]]

		#Run Kohonen algorithm to train
		for i in range(num_iters-1):
			for x in X.T:
				win=winning_center(x,self.weights)
				update_centers(self.weights,win,x,i,self.lattice,num_iters,lattice_distance)
			if i%50==0: print("completed iteration: ",i)
		#Run last iteration separate to store winning weights.
		self.WIN_IND=[]
		for x in X.T:
			win=winning_center(x,self.weights)
			self.WIN_IND.append(win)
			update_centers(self.weights,win,x,num_iters,self.lattice,num_iters,lattice_distance)




	#Give plot of initial confiugration and final configuration (for dimensions 1 and 2)
	def make_plot(self,W,X):
		lat_nm=self.lattice_nm
		params=self.lattice
		if lat_nm=="line":
			plt.figure()
			plt.title("Chosen Lattice")
			x=np.linspace(0,params[1]-1,params[1])
			plt.plot(x,np.full(params[1],0),'-o')
			plt.plot(x,np.full(params[1],0))

			for i in range(params[1]):
				plt.annotate(str(i), (i,0))

		if lat_nm=="rectangle":
			plt.figure()
			plt.title("Chosen Lattice")
			x=np.linspace(0,params[1]-1,params[1])
			y=np.linspace(0,params[2]-1,params[2])

			for a in range(params[1]):
				for b in range(params[2]):
					plt.plot(x[a],y[b],'-o')
			for a in range(params[1]):
				for b in range(params[2]):
					plt.plot(x[a],y[b])

			for i in range(params[1]*params[2]):
			 	plt.annotate(str(i), (i%params[1], np.floor(i/params[1])))

		if lat_nm=="circle":
			plt.figure()
			plt.title("Chosen Lattice")
			t=np.linspace(0,params[1]-1,params[1])
			x=np.cos(t*np.pi*2/params[1])
			y=np.sin(t*np.pi*2/params[1])
			plt.plot(x,y,'-o',color='blue')
			plt.plot(x,y,color='blue')
			plt.plot([1,x[params[1]-1]],[0,y[params[1]-1]],color='blue')

			for i in range(params[1]):
				plt.annotate(str(i), (x[i],y[i]))


		if lat_nm=="torus":
			plt.figure()
			plt.title("Chosen Lattice")
			x=np.linspace(0,params[1]-1,params[1])
			y=np.linspace(0,params[2]-1,params[2])

			for a in range(params[1]):
				for b in range(params[2]):
					plt.plot(x[a],y[b],'-o')
			for a in range(params[1]):
				for b in range(params[2]):
					plt.plot(x[a],y[b])

			for i in range((params[1]-1)*(params[2]-1)):
				coords=lat_coords_torus(self.lattice,i)
				for c in coords.T:
					plt.annotate(str(i), (c[0],c[1]))

		if np.shape(W)[0]==1 or np.shape(W)[0]==2:
			plt.figure()
			plt.title('The Data')
			plt.scatter(X[0,:],X[1,:])

			plt.figure()
			plt.title('Initial configuration')
			plt.scatter(W[0,:],W[1,:])
			n=range(W.shape[1])

			for i, txt in enumerate(n):
				plt.annotate(txt, (W[0,i],W[1,i]))

			plt.figure()
			plt.title('Final Configuration')
			plt.scatter(self.weights[0,:],self.weights[1,:])
			for i, txt in enumerate(n):
				plt.annotate(txt, (self.weights[0,i],self.weights[1,i]))

			plt.figure()
			plt.title('Final Configuration with Data')
			plt.scatter(self.weights[0,:],self.weights[1,:],color='red')
			plt.scatter(X[0,:],X[1,:],color='black')
		



	def mse(self,X):
		#compute mean square error on training set X and center weights (reference vectors)
		ERR=[]
		for cnt, x in enumerate(X.T):
			l=self.WIN_IND[cnt]
			err=la.norm(x-self.weights[:,l],2)**2
			ERR.append(err)
		return sum(ERR)


	#
	def neighbors_score(self):
		return

	def classify(self,Y):
		ERR=[]
		L=[]
		for y in Y.T:
			l=winning_center(y,self.weights)
			err=la.norm(y-self.weights[:,l],2)**2
			L.append(l)
			ERR.append(err)
			mse=sum(ERR)
		return L,ERR,mse


# WIN_IND=[]
# 			if not i%40:
# 				plt.figure(i)
# 				# plt.xlim(0,1)
# 				# plt.ylim(0,1)
# 				ttl='eps='+str(eps)+' sigma='+str(sigma)
# 				plt.title(ttl)
# 				plt.scatter(self.weights[0,:],self.weights[1,:],color='red')
# 				#plt.scatter(W[0,:],W[1,:],color=cmap(i))
# 				#plt.scatter(X[0,:],X[1,:],color='black',marker='x')
# 				plt.savefig(imgpath+'iter'+str(i))

# plt.figure(num_iters)
# 		plt.scatter(X[0,:],X[1,:],color='black',marker='x')
# 		plt.scatter(self.weights[0:1],self.weights[1,:],color='red')
# 		plt.savefig(imgpath+'final')