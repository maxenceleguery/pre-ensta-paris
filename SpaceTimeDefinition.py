import numpy as np

def defineSchw(l):
        """
        Definition of the Schwarzschild space-time

        Parameters
        -----------
        l : int
                Degree of spherical harmonic
        """

        eps=-1 # -1 scalar, 0 electromagnetic, 3 gravitational

        def alpha(n,w):
                return n*n + (2*-1j*w+2)*n + 2*-1j*w + 1

        def beta(n,w):
                return -(2*n*n + (8*-1j*w+2)*n + 8*-1j*w*-1j*w + 4*-1j*w + l*(l+1) - eps)

        def gamma(n,w):
                return n*n + 4*-1j*w*n + 4*-1j*w*-1j*w - eps - 1
        
        return np.array([alpha,beta,gamma])

def defineKerrThetaFrac(alpha,m,A):
        """
        Definition of the Kerr space-time (angular equation)

        Parameters
        -----------
        alpha : float
                Angular parameter
        m : int

        A : complex
                Separation constant
        """
        def alpha_theta(n,w):
                return -2*(n+1)*(n+abs(m)+1)

        def beta_theta(n,w):
                return n*(n-1) + 2*n*(abs(m)+1-2*alpha*w) - (2*alpha*w*(abs(m)+1) - abs(m)*(abs(m)+1)) - (alpha*alpha*w*w + A)

        def gamma_theta(n,w):
                return 2*alpha*w*(n+abs(m))
        
        return np.array([alpha_theta,beta_theta,gamma_theta])

def defineKerrRFrac(alpha,m,A):
        """
        Definition of the Kerr space-time (radial equation)

        Parameters
        -----------
        alpha : float
                Angular parameter
        m : int

        A : complex
                Separation constant
        """
        
        b = np.sqrt(1-4*alpha*alpha)

        def c0(w):
                return 1 - 1j*w - (2j/b)*(w/2 - alpha*m)

        def c1(w):
                return -4 + 2*(2+b)*1j*w + (4j/b)*(w/2 - alpha*m)

        def c2(w):
                return 3 - 3j*w - (2j/b)*(w/2 - alpha*m)

        def c3(w):
                return w*w*(4 + 2*b - alpha*alpha) - 2*alpha*m*w - 1 - A + (2+b)*1j*w + ((4*w+2j)/b)*(w/2 - alpha*m) 

        def c4(w):
                return 1 - 2*w*w - 3j*w - ((4*w+2j)/b)*(w/2 - alpha*m)

        def alpha_r(n,w):
                return n*n + (c0(w) + 1)*n + c0(w)
                
        def beta_r(n,w):
                return -2*n*n + (c1(w) + 2)*n + c3(w)
                        
        def gamma_r(n,w):
                return n*n + (c2(w) - 3)*n + c4(w) - c2(w) + 2
        
        return np.array([alpha_r,beta_r,gamma_r])

def defineTaubNUT(l,C):
        """
        Definition of the Taub-NUT space-time (radial equation)

        Parameters
        ----------
        l : float
                NUT charge
        C : complex
                Casimir
        """
        
        l2=l*l
        q=2
        M=1/2
        rplus=M+np.sqrt(M*M+l2)
        rminus=M-np.sqrt(M*M+l2)

        rplus2=rplus*rplus
        rminus2=rminus*rminus

        s=(rplus2 + l2)/(2*np.sqrt(M*M+l2))
        dr=rplus-rminus
        dr2=dr*dr

        def P(k,w):
                w2 = w*w
                #C=6.5
                #C = -1/4-pow((2*l*w).imag,2) 
                #C = q*(q+1)
                match k:
                        case 0:
                                return w2*(rplus2 + l2*l2 + 2*l2*rplus2)/dr2
                        case 1:    
                                return 1j*w-1 + 1j*w*dr + w2*(-2*rminus*rplus2-4*l2*l2 - 2*l2*rplus*rminus - 4*l2*rplus2)/dr2 + 4*l2*w2 - C
                        case 2:
                                return (3 - 1j*w)*(1-1j*w) - 2*(w2 + 1j*w)*dr - 2*(4*l2*w2 - C) - w2*dr2 + w2*(rminus2*rplus2 + 2*rminus2*rplus +6*l2*l2 + 2*l2*rminus2 + 4*l2*rminus*rplus + 2*l2*rplus2)/dr2
                        case 3:
                                return (2*1j*w - 3)*(1-1j*w) + (2*w2 + 1j*w)*dr + 4*l2*w2 - C + w2*(-2*rminus*rminus2*rplus - 4*l2*l2 - 4*l2*rminus2 - 2*l2*rminus*rplus)/dr2
                        case 4:
                                return (1 - 1j*w)*(1 - 1j*w) + w2*(rminus2*rminus2+l2*l2+2*l2*rminus2)/dr2

        def An1(n,w):
                return P(0,w) +                (n+1 - 1j*w*s)*((n - 1j*w*s) +        1)
        def Bn1(n,w):
                return P(1,w) +                (n - 1j*w*s)*(-4*(n-1 - 1j*w*s) +     (-2*(1-1j*w) + 2*1j*w*dr - 4))
        def Cn1(n,w):
                return P(2,w) +                (n-1 - 1j*w*s)*(6*(n-2 - 1j*w*s) +   (6*(1-1j*w) - 4*1j*w*dr + 6))
        def Dn1(n,w):
                return P(3,w) +                (n-2 - 1j*w*s)*(-4*(n-3 - 1j*w*s) +   (-6*(1-1j*w) + 2*1j*w*dr - 4))
        def En1(n,w):
                return P(4,w) +                (n-3 - 1j*w*s)*( (n-4 - 1j*w*s) +     (2*(1-1j*w) + 1))
        
        return np.array([An1,Bn1,Cn1,Dn1,En1])

def defineTaubNUT2(l,C,m):
        """
        Definition of the Taub-NUT space-time (angular equation)

        Parameters
        ----------
        l : float
                NUT charge
        C : complex
                Casimir
        m : int
        """

        def T(i,j,w):
                omega = 2*l*w
                N=m-omega
                alpha = (m-omega)/2
                beta = (m+omega)/2
                match i:
                        case 0:
                                match j:
                                        case 0:
                                                return alpha*alpha - N*N/4
                                        case 1:
                                                return -2*alpha*alpha - 2*alpha*beta - alpha - beta - N*omega - omega*omega + C
                                        case 2:
                                                return pow(alpha+beta,2) + (alpha+beta) - C
                        case 1:
                                match j:
                                        case 0:
                                                return 2*alpha + 1
                                        case 1:
                                                return -4*alpha - 2*beta - 3 
                                        case 2:
                                                return 2*(alpha + beta + 1)
                        case 2:
                                match j:
                                        case 0:
                                                return 1
                                        case 1:
                                                return -2
                                        case 2:
                                                return 1
                                        
        def A(k,w):
                return T(0,0,w) + (k+1)*T(1,0,w) + (k+1)*k*T(2,0,w)
        def B(k,w):
                return T(0,1,w) + (k)*T(1,1,w) + k*(k-1)*T(2,1,w)
        def D(k,w):
                return T(0,2,w) + (k-1)*T(1,2,w) + (k-1)*(k-2)*T(2,2,w)
        
        return np.array([A,B,D])