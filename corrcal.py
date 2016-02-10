import numpy
import ctypes

mylib=ctypes.cdll.LoadLibrary("libcorrcal.so")
make_curve_part_c=mylib.make_curve_part
make_curve_part_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

update_grad_c=mylib.update_grad
update_grad_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]

inverse_symmetric_c=mylib.inverse_symmetric
inverse_symmetric_c.argtypes=[ctypes.c_void_p,ctypes.c_int]

sparse_Mg_transpose_c=mylib.sparse_Mg_transpose
sparse_Mg_transpose_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

sparse_Mg_transpose_1vec_c=mylib.sparse_Mg_transpose_1vec
sparse_Mg_transpose_1vec_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

test_inv_c=mylib.test_inv
test_inv_c.argtypes=[ctypes.c_void_p,ctypes.c_int]

sparse_inv_c=mylib.sparse_inv
#sparse_inv_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
sparse_inv_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]

sparse_inv2_c=mylib.sparse_inv
#sparse_inv_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
sparse_inv2_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]

add_outer_product_c=mylib.add_outer_product
add_outer_product_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]


diag_mult_c=mylib.diag_mult
diag_mult_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

apply_calib_to_vis_c=mylib.apply_calib_to_vis
apply_calib_to_vis_c.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]






class sparse:
    def __init__(self,diag,vecs,isinv=False):
        self.diag=diag.copy()
        self.vecs=numpy.matrix(vecs.copy())
        self.mat=None
        self.isinv=isinv
    def copy(self):
        return sparse(self.diag,self.vecs,self.isinv)
    def __mul__(self,vec):
        if isinstance(vec,numpy.ndarray):
            vec=numpy.matrix(vec)
        tmp=self.vecs*vec.transpose()
        if self.isinv:
            tmp2=-self.vecs.transpose()*tmp
        else:
            tmp2=self.vecs.transpose()*tmp
        nn=vec.shape
        
        diag_mult_c(self.diag.ctypes.data,vec.ctypes.data,nn[1],nn[0],tmp2.ctypes.data)
        return tmp2
    def mul2(self,vec):
        if isinstance(vec,numpy.ndarray):
            vec=numpy.matrix(vec)
        nn=vec.shape
        tmp=numpy.matrix(numpy.zeros([nn[0],nn[1]]))
        diag_mult_c(self.diag.ctypes.data,vec.ctypes.data,nn[1],nn[0],tmp.ctypes.data)
        tmp2=self.vecs.transpose()*tmp.transpose
        return tmp2
                         
        
        tmp=self.vecs*vec.transpose()
        if self.isinv:
            tmp2=-self.vecs.transpose()*tmp
        else:
            tmp2=self.vecs.transpose()*tmp

        
        diag_mult_c(self.diag.ctypes.data,vec.ctypes.data,nn[1],nn[0],tmp2.ctypes.data)
        return tmp2




    def inv(self):
        val=self.copy()
        val.isinv=not(self.isinv)
        nn=self.vecs.shape
        #val.mat=numpy.matrix(numpy.zeros([nn[0],nn[0]]))
        #sparse_inv_c(self.diag.ctypes.data,self.vecs.ctypes.data,nn[1],nn[0],val.vecs.ctypes.data,val.diag.ctypes.data,val.mat.ctypes.data )
        sparse_inv_c(self.diag.ctypes.data,self.vecs.ctypes.data,nn[1],nn[0],val.vecs.ctypes.data,val.diag.ctypes.data )
        return val
    def inv2(self):
        val=self.copy()
        val.isinv=not(self.isinv)
        nn=self.vecs.shape
        val.mat=numpy.matrix(numpy.zeros(nn[0]))
        sparse_inv2_c(self.diag.ctypes.data,self.vecs.ctypes.data,nn[1],nn[0],val.diag.ctypes.data,val.mat.ctypes.data )
        return val


def inverse_symmetric(mat):
    n=len(mat)
    inverse_symmetric_c(mat.ctypes.data,n)


def redundant_cal(vis,ant1,ant2,ind,projvecs,amp=1.0,thresh=1e-7):
    nant=numpy.max([numpy.max(ant1),numpy.max(ant2)])+1
    g=(numpy.ones(nant))+numpy.complex(0,0)
    finished=False
    vis_cur=vis.copy()

    

    #projvecs=numpy.zeros([4,2*nant]);
    #projvecs[0,0::2]=1;
    #projvecs[1,1::2]=1;
    #projvecs[2,1::2]=xx;
    #projvecs[3,1::2]=yy;
    #projvecs=numpy.matrix(projvecs)
    
    iter=0
    vv=vis[0::2]+numpy.complex(0,1)*vis[1::2]
    while finished==False:
        print 'iter is ' + repr(iter)
        dg,grad,curve=redundant_cal_curve_deriv(vis_cur,ant1,ant2,ind,projvecs,amp)
        #print dg[0,0:5]
        err=numpy.mean(numpy.abs(dg))
        
        print err
        if err<thresh:
            finished=True
        else:
            ddg=dg[0,0::2]+numpy.complex(0,1)*dg[0,1::2]

            g=g*(1+numpy.array(ddg))
            #print g[0,0:5]
            #print ddg[0,0:5]
            tmp=vv.copy()
            apply_calib_to_vis_c(tmp.ctypes.data,tmp.size,g.ctypes.data,ant1.ctypes.data,ant2.ctypes.data)

            vis_cur[0::2]=numpy.real(tmp)
            vis_cur[1::2]=numpy.imag(tmp)
            #print vis_cur[0:5]
        iter=iter+1
        if iter==10:
            finished=True
    return g
        
    
    
def redundant_cal_curve_deriv(vis,ant1,ant2,ind,projvecs,amp=1.0):

    nant=numpy.max([numpy.max(ant1),numpy.max(ant2)])+1
    nn=numpy.shape(ind)
    nblock=nn[0]
    grad=numpy.zeros(2*nant)
    curve=numpy.zeros([2*nant,2*nant])

    nmax=numpy.max(ind[:,1]-ind[:,0])
    vecs=numpy.zeros([2,2*nmax])
    vecs[0,0::2]=amp
    vecs[1,1::2]=amp
    big_tmp=numpy.matrix(numpy.zeros([2*nblock,2*nant]))

    noise_grad=numpy.zeros(2*nant);
    print  'doing gradient'
    for i in range(nblock):
        m=ind[i][1]-ind[i][0]
        a1=ant1[ind[i][0]:ind[i][1]]
        a2=ant2[ind[i][0]:ind[i][1]]
        cn=sparse(numpy.ones(2*m),vecs[:,0:2*m])
        cn_inv=cn.inv()
        myvis=vis[2*ind[i][0]:2*ind[i][1]]
        vv=cn_inv*myvis
        
        grad=grad+sparse_Mg_transpose(vv,myvis,a2,a1,nant)
        if (1):
            update_grad(vv,noise_grad,a1,a2)
        else:
            for ii in range(a1.size):
                f1=(vv[2*ii]**2+vv[2*ii+1]**2)
                f2=(vv[2*ii]**2+vv[2*ii+1]**2)
                noise_grad[2*a1[ii]]=noise_grad[2*a1[ii]]+(f1+f2)
                noise_grad[2*a2[ii]]=noise_grad[2*a2[ii]]+(f1+f2)


        tmp=numpy.matrix(sparse_Mg_transpose(myvis,cn_inv.vecs,a1,a2,nant))
        if 0:
            curve=curve+tmp.transpose()*tmp
        else:
            big_tmp[2*i:2*i+2,:]=tmp
            #nn=tmp.shape
            #add_outer_product_c(curve.ctypes.data,tmp.ctypes.data,nn[1],nn[0])
    print 'making curve'
    curve=make_curve_part(vis,ant1,ant2,nant)-big_tmp.transpose()*big_tmp
    
    #grad=grad-noise_grad

    cc=curve+projvecs.transpose()*projvecs
    print 'inverting'
    minv=inverse_projvected(cc,projvecs)
    dg=grad*minv

    return dg,grad,curve
        

def make_curve_part(vis,ant1,ant2,nant=0):
    if (nant==0):
        a=ant1.max()
        b=ant2.max()
        nant=numpy.max([a,b])+1
    mat=numpy.zeros([2*nant,2*nant])
    make_curve_part_c(vis.ctypes.data,  ant1.ctypes.data,  ant2.ctypes.data,  vis.size/2,  nant,mat.ctypes.data)
    return mat

def update_grad(vv,grad,a1,a2):
    update_grad_c(vv.ctypes.data,grad.ctypes.data,a1.ctypes.data,a2.ctypes.data,a1.size)

def sparse_Mg_transpose(vis,vecs,ant1,ant2,nant=0):
    if (nant==0):
        a=ant1.max()
        b=ant2.max()
        nant=numpy.max([a,b])+1

    asdf=vecs.shape
    if len(asdf)==1:
        nvec=1
    else:
        nvec=asdf[0]

    MTv=numpy.zeros([nvec,2*nant])
    nvis=vis.size/2    
    #print nvec, nant, nvis
    sparse_Mg_transpose_c(vecs.ctypes.data,vis.ctypes.data,ant1.ctypes.data,ant2.ctypes.data,nvis,nant,nvec,MTv.ctypes.data)
    return MTv

def sparse_Mg_transpose_1vec(vis,vec,ant1,ant2,nant=0):
    if (nant==0):
        a=ant1.max()
        b=ant2.max()
        nant=numpy.max([a,b])+1

    MTv=numpy.zeros(2*nant)
    nvis=vis.size/2    
    sparse_Mg_transpose_1vec_c(vec.ctypes.data,vis.ctypes.data,ant1.ctypes.data,ant2.ctypes.data,nvis,nant,MTv.ctypes.data)
    return MTv
def test_inv(mat):
    nn=mat.shape
    assert(nn[0]==nn[1])
    nn=nn[0]
    test_inv_c(mat.ctypes.data,nn)
def sparse_inv(diag,vecs):
    nn=vecs.shape
    if len(nn)==1:
        n=nn[0]
        nvec=1
    else:
        nvec=nn[0];
        n=nn[1];
    #print n,nvec
    diag_out=diag.copy()
    #mat=numpy.matrix(numpy.zeros([nvec,nvec]))
    vecs_inv=numpy.matrix(numpy.zeros([nvec,n]))
    #sparse_inv_c(diag.ctypes.data,vecs.ctypes.data,n,nvec,mat.ctypes.data,vecs_inv.ctypes.data)
    sparse_inv_c(diag.ctypes.data,vecs.ctypes.data,n,nvec,vecs_inv.ctypes.data,diag_out.ctypes.data)
    return vecs_inv,diag_out
def inverse_projvected(mat,vecs):
    #print mat.shape
    #print vecs.shape

    #minv=numpy.linalg.inv(mat)
    minv=mat.copy()
    inverse_symmetric(minv)
    tmp=vecs*minv*vecs.transpose()
    small_mat=numpy.linalg.inv(tmp)
    fwee=vecs*minv
    minv=minv-fwee.transpose()*small_mat*fwee
    return minv

