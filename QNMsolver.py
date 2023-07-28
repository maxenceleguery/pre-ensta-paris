import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib as mlp
import time
import os
import scipy as sc


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

class QNMsolver:
        """Solver of quasinormal modes using Leaver method (continued fraction) based on the coefficients of the recursive relation"""

        def __init__(self,coefs : np.ndarray):
                """Initialize the solver

                Parameters
                ----------
                coefs : ndarray
                        Array with the function defining the recursive relation (must have parameters in order : (n,w))

                """
                self.res = [0,3,0.01]

                self.Xs, self.Ys, self.labels = None, None, None

                self.depth = 40
                self.coefs = coefs # ndarray of the coefficients of recursion (functions of (n,w))
                self.nb_reduction = len(coefs) - 3
                self.new_coefs = np.zeros((len(coefs),self.depth+1,self.nb_reduction+1),dtype=object)

        def setResolution(self,res):
                """Change the range of real and imaginary values of w

                Parameters
                ----------
                res : ndarray
                        [start,end,step]
                """
                self.res = res

        def updateCoefs(self,coefs : np.ndarray):
                """Update the coefficients ndarray of the functions defining the recursion relation
                
                Parameters
                ----------
                coefs : ndarray
                        Array with the function defining the recursive relation
                """
                self.coefs = coefs

        def __recu(self,i,w,n):
                if i==n:
                        return 1
                else:
                        i=i+1
                        return 1E-20 + self.new_coefs[1,i-1,-1] - self.new_coefs[0,i-1,-1]*self.new_coefs[2,i,-1]/self.__recu(i,w,n)

        def precompute(self, w : complex):
                """Precompute the coefficients using Gauss reduction to get a 3-terms recursive relation
                
                Parameters
                ----------
                w : complex
                        Complex number representing the frequency to test 
                """
                for j in range(self.nb_reduction+1):
                        for i in range(self.depth+1):
                                for k in range(len(self.coefs)-j):
                                        if j==0:
                                                self.new_coefs[k,i,j] = self.coefs[k](i,w)
                                        else:
                                                if i>=len(self.coefs)-j-1 and k!=0:
                                                        self.new_coefs[k,i,j] = self.new_coefs[k,i,j-1] - self.new_coefs[k-1,i-1,j]*self.new_coefs[len(self.coefs)-j,i,j-1]/self.new_coefs[len(self.coefs)-j-1,i-1,j]
                                                else:
                                                        self.new_coefs[k,i,j] = self.new_coefs[k,i,j-1]

        @jit
        def continuedFrac(self,w:complex) -> float:
                """Compute the continued fraction for a given complex frequency, the zeros are the QNM frequencies
                
                Parameters
                ----------
                w : complex
                        Complex number
                        
                Return
                ----------
                float 
                        the modulus of the continued fraction evaluated at w
                """
                self.precompute(w)
                return abs(self.__recu(0,w,self.depth))
        
        def addComparaison(self,Xs,Ys,labels=None):
                """Add points to the graph for comparaison
                
                Parameters
                ----------
                Xs : ndarray of ndarray of float
                        Array of the abscisses of values to add for comparaison
                Ys : ndarray of ndarray of float
                        Array of the ordinates of values to add for comparaison
                labels : ndarray of str
                        Labels to display
                """
                self.Xs, self.Ys, self.labels = Xs, Ys, labels
        
        def plot(self,title:str,saveInFolder : str=None,frame_number:int=None):
                """Compute the continued fraction on the complex plane
                
                Parameters
                ----------
                title : str
                        Name of the plot
                saveInFolder : str, optional
                        If given, unable display and save the plot in the folder
                frame_number : int, optional
                        Number of the current frame (useful to build animation)
                """
                real = np.arange(self.res[0],self.res[1],self.res[2])
                imag = np.arange(self.res[0],self.res[1],self.res[2])
                X, _ = np.meshgrid(real, imag)
                Z = np.zeros(X.shape)
                Z2 = np.zeros(X.shape,dtype='complex_')

                print("Start of computation")
                start = time.time()

                for idxR,r in enumerate(real):
                        for idxI,i in enumerate(imag):
                                Z2[idxI,idxR] = r-i*1j
                                #taux=100*(idxR*len(self.imag)+idxI)/(len(self.real)*len(self.imag))
                                #print(f"Avancement : {taux:.2f}%\r", end='')

                Z = self.continuedFrac(Z2)

                end = time.time()

                print(f"Time elapsed : {end-start:0.4f}s")

                cmap = plt.cm.coolwarm

                fig, (ax1) = plt.subplots(ncols=1)
                pc = ax1.pcolormesh(real,imag,Z,norm=mlp.colors.LogNorm(),cmap=cmap)
                fig.colorbar(pc, ax=ax1)
                ax1.set_title(title,fontsize=15)
                ax1.set_xlabel("$\omega_r$",fontsize=15)
                ax1.set_ylabel("$-\omega_i$",fontsize=15)

                if self.labels:
                        if self.Xs != None and self.Ys != None:
                                for x,y,label in zip(self.Xs,self.Ys,self.labels):
                                        plt.scatter(np.array(x),-np.array(y),label=label)
                        plt.legend()
                else:
                        if self.Xs != None and self.Ys != None:
                                for x,y in zip(self.Xs,self.Ys):
                                        plt.scatter(np.array(x),-np.array(y))
                if saveInFolder!=None:
                        path = "./"+saveInFolder
                        if not os.path.exists(path):
                                os.mkdir(path)
                        if frame_number != None:
                                plt.savefig(path+f"/frame{frame_number}.png")
                        else:
                                plt.savefig(path+f"/{title}.png")
                        plt.close()
                else:
                        pass
                        #plt.show()

        def show(self) -> None:
                """Show the Matplotlib figure
                """
                plt.show()

        def __getMin(self,x0:complex) -> complex:
                """Return a root of the continued fraction giving a starting value
                
                Parameters
                ----------
                x0 : complex
                        Initial value to search a root

                Return
                ----------
                        A root of the continued fraction
                        None if the algorithm did not converged
                """
                def func(w):
                        return [self.continuedFrac(w[0]+w[1]*1j),0]
                sol = sc.optimize.root(func, [x0.real,x0.imag])
                if sol.success:
                        return sol.x[0] + 1j*sol.x[1]
                else:
                        return None
        
        def isSameRoot(self,w1:complex,w2:complex) -> bool:
                """Tell if two complex number are closed enough to be considered equal
                
                Parameters
                ----------
                w1 : complex
                        First complex value
                w2 : complex
                        Second complex value
                
                Return
                ----------
                boolean
                        True if closed enough
                """
                return abs(w1.real - w2.real) < 1E-3 and abs(w1.imag - w2.imag) < 1E-3

        def computeQNM(self) -> list:
                """Compute the quasinormal modes of the given space-time \n
                The search is done on the bottom half of the complex plane according to the resolution
                
                Return
                ----------
                list of complex values
                """
                real = np.arange(self.res[0],self.res[1],self.res[2]*10)
                imag = np.arange(self.res[0],self.res[1],self.res[2]*10)
                Z = []
                print("\n")
                for idxR,r in enumerate(real):
                        for idxI,i in enumerate(imag):
                                root = self.__getMin(r-i*1j)
                                if root != None:
                                        Z.append(root)
                                taux=100*(idxR*len(imag)+idxI)/(len(real)*len(imag))
                                print(f"Avancement : {taux:.2f}%    \r", end='')
                print("\n")
                Z2=[]
                for w in Z:
                        found=False
                        for w0 in Z2:
                                if self.isSameRoot(w0,w):
                                        found=True
                                        break
                        if (not found) and abs(w.real)>1E-3 and self.res[0]<w.real<self.res[1] and self.res[0]<-w.imag<self.res[1] and self.continuedFrac(w) < 1E-2:
                                Z2.append(w)
                Z2 = [i for i in Z2 if i is not None]
                print(f"Unique roots found : {len(Z2)}")
                return Z2