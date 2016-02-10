// gcc-4.8 -O3 -shared -fPIC -std=c99 -o libcorrcal.so corrcal.c    
// gcc-4.8 -O3 -shared -fPIC -std=c99 -o libcorrcal.so corrcal.c   -L/Users/sievers/local/src/OpenBLAS -lopenblas -lpthread
// gcc-5 -O3 -shared -fPIC -std=c99 -o libcorrcal.so corrcal.c   -L/Users/sievers/local/src/OpenBLAS -lopenblas -lpthread

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>



#define INV_SYMM
#ifdef INV_SYMM
/*--------------------------------------------------------------------------------*/
void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
int dpotrf(char uplo, int n, double *a, int lda)
{
  int info=0;
  dpotrf_(&uplo,&n,a,&lda,&info);
  if (info)
    printf("info was %d in dpotrf.\n",info);
  return info;
}

/*--------------------------------------------------------------------------------*/
void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
int dpotri(char uplo, int n, double *a, int lda)
{
  int info=0;
  dpotri_(&uplo,&n,a,&lda,&info);
  if (info)
    printf("info was %d in dpotrf.\n",info);
  return info;
}
/*--------------------------------------------------------------------------------*/
void symmetrize_mat(double *a, int n)
{
  for (int i=0;i<n;i++)
    for (int j=i+1;j<n;j++)
      a[j*n+i]=a[i*n+j];
}
/*--------------------------------------------------------------------------------*/

void inverse_symmetric(double *a, int n)
{
  dpotrf('L',n,a,n);
  dpotri('L',n,a,n);
  symmetrize_mat(a,n);
}
/*--------------------------------------------------------------------------------*/


#endif


void make_curve_part(double *vis, int *ant1, int *ant2, int nvis, int nant, double *mat_in)
{
  //printf("hello!  nvis is %d\n",nvis);
  //for (int i=0;i<5;i++)
  // printf("%4d %4d\n",ant1[i],ant2[i]);
  
  //memset(mat_in,0,nant*nant*4*sizeof(double));

  double **mat=(double **)malloc(sizeof(double *)*nant*2);
  mat[0]=mat_in;
  for (int i=1;i<2*nant;i++) {
    mat[i]=mat[i-1]+2*nant;    
  }
  

#pragma omp parallel for
  for (int i=0;i<nvis;i++) {
    //for (int i=0;i<10;i++) {
    int j1=ant1[i];
    int j2=ant2[i];
    double val=vis[2*i]*vis[2*i]+vis[2*i+1]*vis[2*i+1];

    mat[2*j1][2*j1]+=val;
    mat[2*j1+1][2*j1+1]+=val;
    mat[2*j1][2*j2]+=val;
    mat[2*j1+1][2*j2+1]-=val;
    
    mat[2*j2][2*j2]+=val;
    mat[2*j2+1][2*j2+1]+=val;
    mat[2*j2][2*j1]+=val;
    mat[2*j2+1][2*j1+1]-=val;	
  }
  
  free(mat);
}

/*--------------------------------------------------------------------------------*/
void update_grad(double *vv,double *grad, int *a1, int *a2, int len)
{
  for (int ii=0;ii<len;ii++) {
    double f1=vv[2*ii]*vv[2*ii]+vv[2*ii+1]*vv[2*ii+1];
    double f2=vv[2*ii]*vv[2*ii]+vv[2*ii+1]*vv[2*ii+1];
    grad[2*a1[ii]]+=f1+f2;
    grad[2*a2[ii]]+=f1+f2;
  }
}
/*--------------------------------------------------------------------------------*/
void sparse_Mg_transpose(double *vecs_in, double *vis, int *ant1, int *ant2, int nvis, int nant, int nvec, double *MTv_in)
{
  double **MTv=(double **)malloc(sizeof(double *)*nvec);
  MTv[0]=MTv_in;
  for (int i=1;i<nvec;i++) {
    MTv[i]=MTv[i-1]+2*nant;
  }

#if 0
  for (int i=0;i<nvec;i++)
    for (int j=0;j<2*nant;j++)
      MTv[i][j]=0;
#endif

  double **vecs=(double **)malloc(sizeof(double *)*nvec);
  vecs[0]=vecs_in;
  for (int i=1;i<nvec;i++)
    vecs[i]=vecs[i-1]+2*nvis;


  double *d1=(double *)malloc(nvec*sizeof(double));
  double *d2=(double *)malloc(nvec*sizeof(double));
  //printf("nvis,nant, and nvec are %d %d %d\n",nvis,nant,nvec);
  for (int j=0;j<nvis;j++) {
    int i1=2*ant1[j];
    int i2=2*ant2[j];
    //printf("i1 and i2 are %d %d\n",i1,i2);

    for (int i=0;i<nvec;i++) {
      d1[i]=vis[2*j]*vecs[i][2*j] + vis[2*j+1]*vecs[i][2*j+1];
      d2[i]=vis[2*j]*vecs[i][2*j+1] - vis[2*j+1]*vecs[i][2*j];
    }
    
    for (int i=0;i<nvec;i++) {
      MTv[i][i1]+=d1[i];
      MTv[i][i1+1]-=d2[i];
      MTv[i][i2]+=d1[i];
      MTv[i][i2+1]+=d2[i];
    }
    
  }
  
  free(vecs);
  free(d1);
  free(d2);
  free(MTv);
}

/*--------------------------------------------------------------------------------*/

void sparse_Mg_transpose_1vec(double *vec, double *vis, int *ant1, int *ant2, int nvis, int nant, double *MTv)
{
  for (int j=0;j<nvis;j++)
    {
      int i1=2*ant1[j];
      int i2=2*ant2[j];
      double d1=vis[2*j]*vec[2*j]+vis[2*j+1]*vec[2*j+1];
      double d2=vis[2*j]*vec[2*j+1]-vis[2*j+1]*vis[2*j];
      MTv[i1]+=d1;
      MTv[i1+1]-=d2;
      MTv[i2]+=d1;
      MTv[i2+1]+=d2;
    }
}

/*--------------------------------------------------------------------------------*/

void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
void cdpotrf(char uplo, int n, double *a, int lda, int *info)
{
  dpotrf_(&uplo,&n,a,&lda,info);
}

void dpotri_(char *uplo, int *n, double *a, int *lda, int *info);
void cdpotri(char uplo, int n, double *a, int lda, int *info)
{
  dpotri_(&uplo,&n,a,&lda,info);
}


/*--------------------------------------------------------------------------------*/


int  invert_posdef_mat_double(double **mat, int n)
{

  int info;
  cdpotrf('u', n, mat[0], n, &info);
  if (info)
    return info;
  cdpotri('u', n,mat[0], n, &info);
  if (info)
    return info;
  double *mm=mat[0];
  for (int i = 0; i < n; i++)
    for (int j = i+1; j < n; j++) {
      mm[i*n+j]=mm[j*n+i];
    }
  return info;
}
/*--------------------------------------------------------------------------------*/

void dtrtri_(char *uplo, char *diag, int *n, double *a, int *lda, int *info);
void cdtrtri(char uplo, char diag, int n, double *a, int lda, int *info)
{
  dtrtri_(&uplo,&diag,&n,a,&lda,info);
}
int chol_inv(double *mat, int n)
{
  //cholesky factor a symmetric matrix, then invert it
  //on output, mat hold L^-1 (rather than the upper triangle of the full inverse, as calculated by dportri)
  int info=0;
  cdpotrf('u', n, mat, n, &info);
  if (info)
    return info;

  cdtrtri('u','n',n,mat,n,&info);
  if (info)
    return info;
  for (int i=0;i<n;i++)
    for (int j=i+1;j<n;j++)
      mat[i*n+j]=0;
  return info;
  
} 
/*--------------------------------------------------------------------------------*/
void test_inv(double *mat, int n)
{

  int info=invert_posdef_mat_double(&mat,n);
  if (info)
    printf("matrix had a problem, info was %d\n",info);
}
/*--------------------------------------------------------------------------------*/
//void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc, int *info,int len1, int len2);
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc, int *info);

void cdgemm(char transa, char transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc, int *info)
{
  dgemm_(&transa, &transb, &m, &n, &k, &alpha,a,&lda,b,&ldb,&beta,c,&ldc,info);  
}

/*--------------------------------------------------------------------------------*/
void sparse_inv(double *diag, double *vecs, int n, int nvec,  double *ninv_vecs_out, double *diag_inv)
{

  //double *diag_inv=(double *)malloc(n*sizeof(double));
  for (int i=0;i<n;i++)
    diag_inv[i]=1.0/diag[i];

  double *ninv_vecs=(double *)malloc(n*nvec*sizeof(double));
  for (int i=0;i<nvec;i++)
    for (int j=0;j<n;j++)
      ninv_vecs[i*n+j]=vecs[i*n+j]*diag_inv[j];

  double *aa=(double *)malloc(nvec*nvec*sizeof(double));
  int info;
  cdgemm('t','n',nvec,nvec,n,1.0,vecs,n,ninv_vecs,n,0.0,aa,nvec,&info);

  for (int i=0;i<nvec;i++)
    aa[i*nvec+i]++;


  //invert_posdef_mat_double(&aa,nvec);
  chol_inv(aa,nvec);

  cdgemm('n','n',n,nvec,nvec,1.0,ninv_vecs,n,aa,nvec,0.0,ninv_vecs_out,n,&info);

  free(aa);
  //free(diag_inv);
  free(ninv_vecs);
}
/*--------------------------------------------------------------------------------*/
void diag_mult(double *diag, double *vecs, int n, int nvec, double *vecs_out)
{
  for (int i=0;i<nvec;i++)
    for (int j=0;j<n;j++) 
      vecs_out[i*nvec+j]+=diag[j]*vecs[i*nvec+j];
  
}

/*--------------------------------------------------------------------------------*/
void add_outer_product(double *mat,double *vecs, int n, int nvec)
{
  //printf("n and nvec are %d %d\n",n,nvec);
  for (int i=0;i<n;i++)
    for (int j=i;j<n;j++) 
      for (int k=0;k<nvec;k++) {
	mat[i*n+j]+=vecs[k*n+i]*vecs[k*n+j];
	//mat[i*n+j]+=1.0;
      }
  
}

/*--------------------------------------------------------------------------------*/
void apply_calib_to_vis(double complex *vis, int nvis, double complex *g, int *ant1, int *ant2)
{
  for (int i=0;i<nvis;i++) {
    vis[i]/=(conj(g[ant1[i]])*g[ant2[i]]);
  }
}
