#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define SIZE_MAT 512
#define BLOCK_SIZE 32

#define HELP "El programa puede recibir 1 parametro N (tamano de matriz)\
o 2 parametros N y BS (Tamano de bloque para la multiplicacion por bloque)"

int N = SIZE_MAT,  BS = BLOCK_SIZE;

/* Time in seconds from some point in the past */
double dwalltime();

double metrics(double *matriz, double *max, double *min, double *prom) {
    *max = matriz[0];
    *min = matriz[0];
    *prom = 0.0;
    for (int i = 0; i < N * N; i++) {
        if (matriz[i] > *max) {
        *max = matriz[i];
        }
        if (matriz[i] < *min) {
        *min = matriz[i];
        }
        *prom += matriz[i];
    }
    *prom /= N * N;
}

void blkmul(double *ablk, double *bblk, double *cblk, int bs)
{
  int i, j, k; 

  for (i = 0; i < bs; i++)
  {
    for (j = 0; j < bs; j++)
    {
      for  (k = 0; k < bs; k++)
      {
        cblk[i*N + j] += ablk[i*N + k] * bblk[j*N + k];
      }
    }
  }
}

void productoMatriz(double *a, double *b, double *c, int bs)
{
  int i, j, k;

  for (i = 0; i < N; i += bs)
  {
    for (j = 0; j < N; j += bs)
    {
      for  (k = 0; k < N; k += bs)
      {
        blkmul(&a[i*N + k], &b[j*N + k], &c[i*N + j], bs);
      }
    }
  }
}

int main(int argc, char *argv[]) {

  if (argc > 1 && (N = atoi(argv[1])) < 0) {
    printf("parametro N no valido\n");
    printf(HELP);
    exit(1);
  }

  if (argc = 2 && (BS = atoi(argv[2])) < 0) {
    printf("Parametro BS no valido\n");
    printf(HELP);
    exit(1);
  }
  // Declarar punteros a la matriz y reservar espacio
  double time;
  double *matrizA, *matrizB, *matrizC, *matrizD, *matrizR;
    double maxA, minA, promA;
    double maxB, minB, promB;

  matrizA = malloc(N * N * sizeof(double));
  matrizB = malloc(N * N * sizeof(double));
  matrizC = malloc(N * N * sizeof(double));
  matrizD = malloc(N * N * sizeof(double));
  matrizR = malloc(N * N * sizeof(double));

  // Llenar las matrices con valores determinados para pruebas uniformes
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      matrizA[i*N+j] = 1.0;
      matrizB[j*N+i] = 2.0;
      matrizC[i*N+j] = 3.0;
      matrizD[j*N+i] = 4.0;
      matrizR[i*N+j] = 0;
    }
  } 

  time = dwalltime();
  
  // Calcular maximo, minimo y promedio
  metrics(matrizA, &maxA, &minA, &promA);
  metrics(matrizB, &maxB, &minB, &promB);

  // Calcular los productos en R
  productoMatriz(matrizA, matrizB, matrizR, BS);
  double factor = (maxA * maxB - minA * minB) / (promA * promB);
  for (int i = 0; i < N * N; i++) {
    matrizR[i] *= factor;
  }
  
  // Calcular el producto con CxD y realizar la suma final
  productoMatriz(matrizC, matrizD, matrizR, BS);

  // Información de ejecución
  printf("Tiempo: %f\n", dwalltime() - time);
  printf("Matriz de %dx%d\n", N, N);
  printf("Tamano de bloque %d\n", BS);
  
  // Liberar memoria de las matrices
  free(matrizA);
  free(matrizB);
  free(matrizC);
  free(matrizD);
  free(matrizR);

  return 0;
}

#include <sys/time.h>

double dwalltime()
{
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}