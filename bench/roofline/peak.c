// Single-core FMA peak and read bandwidth; run pinned: taskset -c 2 ./peak
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
__attribute__((noinline)) double fma_peak(long iters){
  __m256d a[12]; __m256d m=_mm256_set1_pd(0.999999), c=_mm256_set1_pd(1e-9);
  for(int i=0;i<12;i++) a[i]=_mm256_set1_pd(i);
  for(long k=0;k<iters;k++){ for(int i=0;i<12;i++) a[i]=_mm256_fmadd_pd(a[i],m,c); }
  __m256d s=a[0]; for(int i=1;i<12;i++) s=_mm256_add_pd(s,a[i]); double o[4]; _mm256_storeu_pd(o,s); return o[0];
}
__attribute__((noinline)) double readbw(const double *x, long n, int reps){
  // 8 accumulators: enough to cover the 3-cycle add latency at 2 loads per cycle
  __m256d s[8]; for(int k=0;k<8;k++) s[k]=_mm256_setzero_pd();
  for(int r=0;r<reps;r++) for(long i=0;i<n;i+=32) for(int k=0;k<8;k++) s[k]=_mm256_add_pd(s[k],_mm256_load_pd(x+i+4*k));
  __m256d s0=s[0]; for(int k=1;k<8;k++) s0=_mm256_add_pd(s0,s[k]); double o[4]; _mm256_storeu_pd(o,s0); return o[0];
}
int main(){
  long it=200000000; double t=now(); double r=fma_peak(it); t=now()-t;
  printf("FMA peak: %.1f GFLOP/s (sink %g)\n", it*12.0*4*2/t/1e9, r);
  long sizes[]={16<<10, 256<<10, 4<<20, 16<<20, 512<<20};
  for(int k=0;k<5;k++){ long bytes=sizes[k], n=bytes/8; double *x=aligned_alloc(64,bytes); for(long i=0;i<n;i++) x[i]=1;
    int reps = (int)(4e9/bytes); if(reps<2) reps=2; readbw(x,n,1);
    t=now(); r=readbw(x,n,reps); t=now()-t; printf("read %8ld KB: %6.1f GB/s\n", bytes>>10, (double)bytes*reps/t/1e9); free(x);}
}
