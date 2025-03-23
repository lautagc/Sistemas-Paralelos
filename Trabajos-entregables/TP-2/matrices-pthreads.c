#include <math.h>
#include <time.h>
#include <stdio.h>
#include <float.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <float.h>
#include <semaphore.h>

/* Time in seconds from some point in the past */
double dwalltime();

/* Parsear argumentos */
int parse_args(int argc, char **argv, int *N, int *B, int *T);

/* Inicializa las matrices con valores fijos para pruebas uniformes */
void init_matrices(double *matrizA,  double *matrizB, double *matrizC,
    double *matrizD, double *matrizR);

/* Obtiene el minimo, máximo y promedio de una matriz */
double metrics(double *matriz, double *max, double *min, double *prom, int id, sem_t mutex);

/* Realiza la multiplicación por bloques */
void blkmul(double *ablk, double *bblk, double *cblk, int bs);

/* Realiza el producto y suma de las matrices */
void producto_matriz(double *a, double *b, double *c, int bs, int id);

/* Realiza la multipliación escalar de la matriz */
void producto_escalar(double *matriz, double factor, int id);

/* Validación de los resultados finales */
bool verify_results(double *matrizR);

void *compute_thread(void *ptr);

int N, T, S, BS;
double factor;
double maxA, minA, promA;
double maxB, minB, promB;
double *matrizA, *matrizB, *matrizC, *matrizD, *matrizR;
pthread_barrier_t barrier[2];
sem_t mutexA, mutexB;

int main(int argc, char *argv[])
{

    // Parsear argumentos
    parse_args(argc, argv, &N, &BS, &T);
    S = N * N;

    // Declarar punteros a la matriz y reservar espacio

    matrizA = malloc(N * N * sizeof(double));
    matrizB = malloc(N * N * sizeof(double));
    matrizC = malloc(N * N * sizeof(double));
    matrizD = malloc(N * N * sizeof(double));
    matrizR = malloc(N * N * sizeof(double));

    maxA = DBL_MIN; minA = DBL_MAX; promA = 0;
    maxB = DBL_MIN; minB = DBL_MAX; promB = 0;
    init_matrices(matrizA, matrizB, matrizC, matrizD, matrizR);


    int ids[T];
    pthread_t threads[T];
    pthread_attr_t attr;
    sem_init(&mutexA, 0, 1);
    sem_init(&mutexB, 0, 1);
    pthread_barrier_init(&barrier[0], NULL, T);
    pthread_barrier_init(&barrier[1], NULL, T);

    double time = dwalltime();
    for (int i = 0; i < T; i++) {
        ids[i] = i;
        pthread_create(&threads[i], &attr, compute_thread, &ids[i]);
    }
    
    int *status;
    for (int i = 0; i < T; i++) {
        pthread_join(threads[i], (void*) &status);
    }

    // Información de ejecución
    printf("Tiempo: %f\n", dwalltime() - time);
    printf("Matriz de %dx%d\n", N, N);
    printf("Cantidad de hilos %d\n", T);
    printf("Largo de bloque %d\n", BS);

    // Verificar resultados
    if (verify_results(matrizR)) {
        printf("Los resultados son correctos\n");
    }
    else {
        printf("Los resultados son incorrectos\n");
    }
    
    // Liberar memoria
    free(matrizA);
    free(matrizB);
    free(matrizC);
    free(matrizD);
    free(matrizR);
    sem_destroy(&mutexA);
    sem_destroy(&mutexB);
    pthread_barrier_destroy(&barrier[0]);
    pthread_barrier_destroy(&barrier[1]);

    return 0;
}

void *compute_thread(void *ptr)
{
    int id = *((int *) ptr);

    // Calcular maximo, minimo y promedio
    metrics(matrizA, &maxA, &minA, &promA, id, mutexA);
    metrics(matrizB, &maxB, &minB, &promB, id, mutexB);

    // Calcular los productos en R
    producto_matriz(matrizA, matrizB, matrizR, BS, id);

    //Barrera 1
    pthread_barrier_wait(&barrier[0]);

    factor = (maxA * maxB - minA * minB) / (promA * promB);
    
    producto_escalar(matrizR, factor, id);
    
    //Barrera 2
    pthread_barrier_wait(&barrier[1]);

    // Calcular el producto con CxD y realizar la suma final
    producto_matriz(matrizC, matrizD, matrizR, BS, id);

    pthread_exit((void *) ptr);
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

void init_matrices(double *matrizA,  double *matrizB, double *matrizC,
    double *matrizD, double *matrizR)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++) 
        {
            matrizA[i*N+j] = 1.0;
            matrizB[j*N+i] = 1.0;
            matrizC[i*N+j] = 1.0;
            matrizD[j*N+i] = 1.0;
            matrizR[i*N+j] = 0;
        }
    }
    /*
    Con estos valores la multiplicación de cada celda es N, luego:
    (maxA * maxB - minA * minB) / (promA * promB) = (3 * 1 - (-1) * 1) / (1 * 1) = 4
    Entonces el valor final del cálculo de R es N*5 en cada celda
    */
    matrizA[0] = 3.0; matrizA[1] = -1.0;
}

bool verify_results(double *matrizR)
{
    bool check = true; int row;
    for (int i = 0; i < N; i++)
    {
        row = i * N;
        for (int j = 0; j < N; j++)
        {
            check = check && (matrizR[row+j] == N*5);
        }
    }
    return check;
}

double metrics(double *m, double *max, double *min, double *prom, int id, sem_t mutex) {
    int start, end, lenght;
    lenght = S / T;
    start = id * lenght;
    end = (id != T-1) ? start + lenght : S;

    double local_min = DBL_MAX, local_max = DBL_MIN, local_sum = 0;
    for (int i = start ; i < end; i++) {
        if (m[i] < local_min) local_min = m[i];
        if (m[i] > local_max) local_max = m[i];
        local_sum += m[i];
    }

    local_sum = local_sum / S;
    sem_wait(&mutex);
    if (local_min < *min) *min = local_min;
    if (local_max > *max) *max = local_max;
    *prom = *prom + local_sum;
    sem_post(&mutex);
}

void blkmul(double *ablk, double *bblk, double *cblk, int bs)
{
    int i, j, k, row, col; 

    for (i = 0; i < bs; i++)
    {
        row = i * N;
        for (j = 0; j < bs; j++)
        {
            col = j * N;
            for  (k = 0; k < bs; k++)
            {
                cblk[row + j] += ablk[row + k] * bblk[col + k];
            }
        }
    }
}

void producto_matriz(double *a, double *b, double *c, int bs, int id)
{
    int start, end, lenght;
    lenght = N  / T;
    start = id * lenght;
    end = (id != T-1) ? start + lenght : N;

    int row, col;
    for (int i = start; i < end; i += bs)
    {
        row = i * N;
        for (int j = 0; j < N; j += bs)
        {
            col = j * N;
            for  (int k = 0; k < N; k += bs)
            {
                blkmul(&a[row + k], &b[col + k], &c[row + j], bs);
            }
        }
    }
}

void producto_escalar(double *matriz, double factor, int id)
{
    int start, end, lenght;
    lenght = S / T;
    start = id * lenght;
    end = (id != T-1) ? start + lenght : S;

    for (int i = start; i < end; i++) {
        matriz[i] *= factor;
    }
}

int parse_args(int argc, char **argv, int *N, int *B, int *T) {
    // Configuration OPT
    extern char *optarg;
    extern int  opterr;
    opterr = 0; //Supress getopt errors

    const char *OPT_STRING = "n:b:t:h";
    const int DEFAULT_OPT_ARG_N = 512;
    const int DEFAULT_OPT_ARG_B = 32;
    const int DEFAULT_OPT_ARG_T = 4;
    const char *HELP_MESSAGE = 
        "Este programa realiza ((maxA * maxB - minA * minB) / promA * promB) x [A x B] + [C x D]:\n"
        "-n, --size <number+>: define el largo de la matriz. Por defecto es 512\n"
        "-b, --block-size <number+>: define el largo del bloque para el producto por bloques. Por defecto es 32 \n"
        "-t, --threads <number+>: define la cantidad de hilos a utilizar. Por defecto es 4\n"
    ;

    static struct option long_options[] = {
        {"size", required_argument, 0, 'n'},
        {"block-size", required_argument, 0, 'b'},
        {"threads", required_argument, 0, 't'},
        {"help", no_argument, 0, 'h'},
    };

    // OPT Proccessing
    *N = DEFAULT_OPT_ARG_N;
    *B = DEFAULT_OPT_ARG_B;
    *T = DEFAULT_OPT_ARG_T;

    int opt, arg;
    while ( (opt = getopt_long(argc, argv, OPT_STRING, long_options, NULL)) != -1 ) {
        switch (opt)
        {
        case 'n':
            arg = atoi(optarg);
            *N = (arg > 0) ? arg: DEFAULT_OPT_ARG_N;
            break;

        case 'b':
            arg = atoi(optarg);
            *B = (arg > 0) ? arg: DEFAULT_OPT_ARG_B;
            break;

        case 't':
            arg = atoi(optarg);
            *T = (arg > 0) ? arg: DEFAULT_OPT_ARG_T;
            break;
        
        case 'h':
            printf("%s", HELP_MESSAGE);
            exit(0);
            break;
        
        default: /* ? */
            fprintf(stderr, "%s", HELP_MESSAGE);
            exit(1);
            break;
        }
    }
}
