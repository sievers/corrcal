import numpy
import corrcal
import time
f=open('../corrcal_test.dat')
fwee=numpy.fromfile(f,numpy.int32,3)
nant=fwee[0]
print nant
nunique=fwee[1]
nvis=fwee[2]
xx=numpy.fromfile(f,numpy.float64,nant);
yy=numpy.fromfile(f,numpy.float64,nant);



flub=numpy.fromfile(f,numpy.int32,nunique*2)
aa=flub[0::2]
bb=flub[1::2]
inds=numpy.zeros([nunique,2],numpy.int32)
inds[:,0]=aa-1
inds[:,1]=bb

ant1=numpy.fromfile(f,numpy.int32,nvis/2)-1
ant2=numpy.fromfile(f,numpy.int32,nvis/2)-1

vis=numpy.fromfile(f,numpy.float64,nvis)


f.close()


projvecs=numpy.zeros([4,2*nant]);
projvecs[0,0::2]=1;
projvecs[1,1::2]=1;
projvecs[2,1::2]=xx;
projvecs[3,1::2]=yy;
projvecs=numpy.matrix(projvecs)

g=numpy.ones(nant)


aa=time.time();dg,grad,curve=corrcal.redundant_cal_curve_deriv(vis,ant1,ant2,inds,projvecs,1e0);bb=time.time();print bb-aa

aa=time.time();g=corrcal.redundant_cal(vis,ant1,ant2,inds,projvecs,1e2);bb=time.time();print bb-aa
assert(1==0)

#mat=corrcal.make_curve_part(vis,ant1,ant2)


#cc=curve+projvecs.transpose()*projvecs
#minv=corrcal.inverse_projvected(cc,projvecs)

#dg=grad*minv
print dg[0,0:5]

assert(1==0)

flub=numpy.zeros([2,vis.size]);flub[0,:]=vis;flub[1,:]=2*vis
mm=corrcal.sparse_Mg_transpose(vis,flub,ant1,ant2)
mm2=corrcal.sparse_Mg_transpose_1vec(vis,vis,ant1,ant2)

nn=inds.shape[0]
vecs=numpy.zeros([2*nn,nvis])

for i in range(inds.shape[0]):
    vecs[2*i,2*inds[i][0]:2*inds[i][1]:2]=1
    vecs[2*i+1,2*inds[i][0]+1:2*inds[i][1]:2]=1
amp=1.0
cn=corrcal.sparse(numpy.ones(nvis)*amp,vecs)

cninv=cn.inv()
vv=cninv*vis

grad=corrcal.sparse_Mg_transpose(vv,vis,ant2,ant1)
tmp=corrcal.sparse_Mg_transpose(vis,vecs,ant2,ant1)

