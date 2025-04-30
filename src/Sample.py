# SPDX-License-Identifier: MPL-2.0
from cmath import acos
import numpy as np
from DMCpy import _tools
import h5py as hdf
from DMCpy import TasUBlibDEG
import warnings

def cosd(x):
    return np.cos(np.deg2rad(x))

def sind(x):
    return np.sin(np.deg2rad(x))

def camelCase(string,split='_'):
    """Convert string to camel case from <split> seperated"""

    if not split in string:
        return string
    splitString = string.split(split)
    first = splitString[0]
    others = [x.title() for x in splitString[1:]]
    
    combi = [first]+others
    return ''.join([str(x) for x in combi])

class Sample(object):
    """Sample object to store all information of the sample from the experiment"""
    @_tools.KwargChecker()
    def __init__(self,a=1.0,b=1.0,c=1.0,alpha=90,beta=90,gamma=90,sample=None,name='Unknown',projectionVector1=None, projectionVector2 = None):

        
        if isinstance(sample,hdf._hl.group.Group):
            self.name = np.array(sample.get('name'))[0].decode()
            if self.name is None or self.name == '':
                self.name = 'Unknown'

            unitCell = np.array(sample.get('unit_cell'))

            if not sample.get('unit_cell') is None:
                self.unitCell = unitCell
            else:
                self.unitCell = [1,1,1,90,90,90]
            self.UB = self.B
            
            
        elif np.all([a is not None,b is not None, c is not None]):
            self.unitCell = np.array([a,b,c,alpha,beta,gamma])
           
            self.polarAngle = np.array(None)
            self.rotationAngle = np.array(0)
            self.name=name
            
            r1 = projectionVector1
            r2 = projectionVector2
            self.plane_vector1 = r1
            self.plane_vector2 = r2

            self.planeNormal = np.cross(self.plane_vector1[:3],self.plane_vector2[:3])
            self.UB = self.B
            
        else:
            print(sample)
            print(a,b,c,alpha,beta,gamma)
            raise AttributeError('Sample not understood')

            
    @property
    def unitCell(self):
        return self._unitCell

    @unitCell.getter
    def unitCell(self):
        return np.array([self.a,self.b,self.c,self.alpha,self.beta,self.gamma])#self._unitCell

    @unitCell.setter
    def unitCell(self,unitCell):
        self._unitCell = unitCell
        self.a = unitCell[0]
        self.b = unitCell[1]
        self.c = unitCell[2]
        self.alpha = unitCell[3]
        self.beta  = unitCell[4]
        self.gamma = unitCell[5]
        self.updateCell()
        
    @property
    def a(self):
        return self._a

    @a.getter
    def a(self):
        return self._a

    @a.setter
    def a(self,a):
        if a>0:
            self._a = a
        else:
            raise AttributeError('Negative or null given for lattice parameter a')

    @property
    def b(self):
        return self._b

    @b.getter
    def b(self):
        return self._b

    @b.setter
    def b(self,b):
        if b>0:
            self._b = b
        else:
            raise AttributeError('Negative or null given for lattice parameter b')

    @property
    def c(self):
        return self._c

    @c.getter
    def c(self):
        return self._c

    @c.setter
    def c(self,c):
        if c>0:
            self._c = c
        else:
            raise AttributeError('Negative or null given for lattice parameter c')


    @property
    def alpha(self):
        return self._alpha

    @alpha.getter
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self,alpha):
        if alpha>0 and alpha<180:
            self._alpha = alpha
        else:
            raise AttributeError('Negative,null or above 180 degrees given for lattice parameter alpha')

    @property
    def beta(self):
        return self._beta

    @beta.getter
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self,beta):
        if beta>0 and beta<180:
            self._beta = beta
        else:
            raise AttributeError('Negative,null or above 180 degrees given for lattice parameter beta')

    @property
    def gamma(self):
        return self._gamma

    @gamma.getter
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self,gamma):
        if gamma>0 and gamma<180:
            self._gamma = gamma
        else:
            raise AttributeError('Negative,null or above 180 degrees given for lattice parameter gamma')

    @property
    def UBInv(self):
        return np.linalg.inv(self.UB)


    def updateCell(self):
        self.fullCell = TasUBlibDEG.calcCell(self.unitCell)
        if hasattr(self,'B'): # Update already existing B and UB matrices
            oldBInv = np.linalg.inv(self.B)
            self.B = TasUBlibDEG.calculateBMatrix(self.fullCell)
            newUB = np.dot(np.dot(self.UB,oldBInv),self.B)
            self.UB = newUB
            warnings.warn('Updating UB matrix with B')
        else:
            self.B = TasUBlibDEG.calculateBMatrix(self.fullCell)
            self.UB = self.B

    def saveToHdf(self,entry):
        entry.create_dataset('name',data = [np.string_(self.name)])
        if hasattr(self,'unitCell'):
            entry.create_dataset('unit_cell',data = self.unitCell)

        
    def defineUB(self,HKL1,HKL2,Q1,Q2):
        self.P1 = _tools.LengthOrder(HKL1)
        self.P3 = _tools.LengthOrder(np.cross(HKL1,HKL2))
        self.P2 = _tools.LengthOrder(np.cross(self.P3,HKL1))

        self.projectionVectors = np.array([self.P1,self.P2,self.P3]).T

        
        axisVectors = np.eye(3)
        ## Assume that Q1/HKL1 is along x-axis

        Alpha1 = np.rad2deg(np.arccos(np.dot(Q1,axisVectors[0])/(np.linalg.norm(Q1))))
        Rot1 = np.cross(Q1,axisVectors[0])
        Rot1*=1.0/np.linalg.norm(Rot1)
        ROT1 = _tools.rotMatrix(Rot1,Alpha1)

        # Rotate Q2 into Q1's frame
        Q2Rot = np.dot(ROT1,Q2)
        Q2Rot-= np.dot(axisVectors[0],Q2Rot)*axisVectors[0]# project out [1,0,0] as this rotation has been done by Q1

        Alpha2 = np.rad2deg(np.arccos(np.dot(Q2Rot,axisVectors[1])/(np.linalg.norm(Q2Rot))))#np.rad2deg(np.arccos(Q2Rot[1]/np.linalg.norm(Q2Rot)))
        Rot2 = np.cross(Q2Rot,axisVectors[1])
        Rot2*=1.0/np.linalg.norm(Rot2)
        ROT2 = _tools.rotMatrix(Rot2,Alpha2)

        self.ROT = np.dot(ROT2,ROT1)

        
        self.projectionB = np.diag(np.linalg.norm(np.dot(self.projectionVectors.T,self.B),axis=1))

        # Rotates into the scattering plane
        self.UB = np.dot(self.ROT.T,np.dot(self.projectionB,np.linalg.inv(self.projectionVectors)))#np.linalg.inv(np.dot(Binverse,self.ROT))

        
    def tr(self,proj0,proj1,proj2=None):
        """Convert from projX, projY coordinate to Qx',QY' coordinate."""
        if proj2 is None:
            p0, p1 = np.asarray(proj0), np.asarray(proj1)
            P = np.array([p0,p1])

            projections = np.delete(self.projectionVectors,2,axis=1)
            
            pm = np.delete(np.eye(3),2,axis=0)
            
            convert = np.dot(pm,np.dot(self.ROT,np.dot(self.UB,projections)))
        else:

            p0, p1, p2 = np.asarray(proj0), np.asarray(proj1), np.asarray(proj2)
            P = np.array([p0,p1,p2])

            # permutation order of the projection vectors
            order = np.array([0,1,2])

            projections = self.projectionVectors[:,order]#
            
            convert = np.dot(self.ROT,np.dot(self.UB,projections))
        return np.einsum('ij,j...->i...',convert,P)


    def inv_tr(self,qx,qy, qz = None):
        """Convert from projX, projY coordinate to Qx',QY' coordinate."""
        if qz is None:

            p0, p1 = np.asarray(qx), np.asarray(qy)
            P = np.array([p0,p1])
            projections = np.delete(self.projectionVectors,2,axis=1)
            
            pm = np.delete(np.eye(3),2,axis=0)

            convert = np.linalg.inv(np.dot(pm,np.dot(self.ROT,np.dot(self.UB,projections))))
        else:
            p0, p1, p2 = np.asarray(qx), np.asarray(qy), np.asarray(qz)
            P = np.array([p0,p1,p2])

            # permutation order of the projection vectors
            order = np.array([0,1,2])

            projections = self.projectionVectors[:,order]
            
            convert = np.linalg.inv(np.dot(self.ROT,np.dot(self.UB,projections)))
        
        
        return np.einsum('ij,j...->i...',convert,P)


    def projectionAngle(self):
        V1 = self.tr(1,0)
        V2 = self.tr(0,1)
        return _tools.vectorAngle(V1, V2)

    def format_coord(self,x,y,z=None):
        """Format coordinates from Qx'Qy' in rotated frame into HKL."""
        order = np.array([0,1,2])
        if z is None:
            proj0,proj1 = self.inv_tr(x,y)
            projections = np.delete(self.projectionVectors,2,axis=1)
            rlu = proj0*projections[:,0]+proj1*projections[:,1]
        else:
            proj0,proj1,proj2=self.inv_tr(*np.asarray([x,y,z])[order])
            
            
            projections = self.projectionVectors
            rlu = proj0*projections[:,0]+proj1*projections[:,1]+proj2*projections[:,2]
        return "h = {0:.3f}, k = {1:.3f}, l = {2:.3f}".format(rlu[0],rlu[1],rlu[2])

    
    def calculateQxQyQzToHKL(self,x,y,z):
        """convert from Qx,Qy to HKL."""
        pos = np.array([x,y,z])
        return np.einsum('ij,j...->i...',self.UBInv,pos)

    def calculateHKLToQxQyQz(self,H,K,L):
        """convert HKL to Qx,Qy."""
        pos = np.array([H,K,L])
        return np.einsum('ij,j...->i...',self.UB,pos)

    def calculateHKLtoProjection(self,H,K,L):
        """convert from projections to HKL."""
        #QxQyQz = np.dot(self.ROT,self.calculateHKLToQxQyQz(H,K,L))
        #projection = self.inv_tr(*QxQyQz)
        projection = np.dot(np.linalg.inv(self.projectionVectors),np.array([H,K,L]))
        return projection
    
    def setProjectionVectors(self,p1,p2,p3=None):
        """Set or update the projection vectors used for the View3D
        
        Args:

            - p1 (list): New primary projection, in HKL

            - p2 (list): New secondary projection, in HKL

        Kwargs:

            - p3 (list): New tertiary projection, in HKL. If None, orthogonal to p1 and p2 (default None)
        """
        if not hasattr(self,'UB'):
            raise AttributeError('No UB matrix present in sample.')
        if p3 is None:
            p3 = _tools.LengthOrder(np.dot(np.linalg.inv(self.B),np.cross(np.dot(self.B,p1),np.dot(self.B,p2))))
        
        self.P1=np.array(p1)
        self.P2=np.array(p2)
        self.P3=np.array(p3)
        self.projectionVectors = np.array([self.P1,self.P2,self.P3]).T

        points = [np.dot(self.UB,v) for v in np.asarray([[0.0,0,0.0],p1,p2])]
        rot,tr = _tools.calculateRotationMatrixAndOffset2(points)
        self.ROT = rot