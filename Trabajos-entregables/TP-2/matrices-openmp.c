#include <stdbool.h>
#include <stdlib.h>
#include <getopt.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* Time in seconds from some point in the past */
double dwalltime();

/* Inicializa las matrices con valores fijos para pruebas uniformes */
void init_matrices(double *matrizA,  double *matrizB, double *matrizC,
    double *matrizD, double *matrizR);

/* Parsear argumentos */
int parse_args(int argc, char **argv, int *N, int *B, int *T);

/* Realiza la multiplicación por bloques */
void blkmul(double *ablk, double *bblk, double *cblk, int bs);

/* Validación de los resultados finales */
bool verify_results(double *matrizR);


int N, T, BS, S;

int main(int argc, char *argv[])
{
    // Parsear argumentos
    parse_args(argc, argv, &N, &BS, &T);
    S = N*N;

    // Declarar punteros a la matriz y reservar espacio
    double time;
    double *matrizA, *matrizB, *matrizC, *matrizD, *matrizR;
    int i, j, k;
    double factor;

    matrizA = malloc(N * N * sizeof(double));
    matrizB = malloc(N * N * sizeof(double));
    matrizC = malloc(N * N * sizeof(double));
    matrizD = malloc(N * N * sizeof(double));
    matrizR = malloc(N * N * sizeof(double));

    // Llenar las matrices con valores determinados para pruebas uniformes
    init_matrices(matrizA, matrizB, matrizC, matrizD, matrizR);
    double maxA = DBL_MIN, minA = DBL_MAX, promA = 0;
    double maxB = DBL_MIN, minB = DBL_MAX, promB = 0;

    time = dwalltime();
    
    // Calcular maximo, minimo y promedio
    #pragma omp parallel private (i, j, k) num_threads(T)
    {
        // Calcula AxB
        #pragma omp for nowait schedule(static)
        for (i = 0; i < N; i += BS)
        {
            for (j = 0; j < N; j += BS)
            {
                for  (k = 0; k < N; k += BS)
                {
                    blkmul(&matrizA[i*N + k], &matrizB[j*N + k], &matrizR[i*N + j], BS);
                }
            }
        }

        // Calcula minimo, maximo y promedio de A
        #pragma omp for reduction(min: minA) reduction(max: maxA) reduction(+:promA) schedule(static)
        for (int i = 0; i < S; i++) {
            if (matrizA[i] > maxA) {
                maxA = matrizA[i];
            }
            if (matrizA[i] < minA) {
                minA = matrizA[i];
            }
            promA += matrizA[i];
        }

        // Calcula minimo, maximo y promedio de B
        #pragma omp for reduction(min: minB) reduction(max: maxB) reduction(+:promB) schedule(static)
        for (i = 0; i < S; i++) {
            if (matrizB[i] > maxB) {
                maxB = matrizB[i];
            }
            if (matrizB[i] < minB) {
                minB = matrizB[i];
            }
            promB += matrizB[i];
        }

        // Un único hilo calcula los promedios y el factor
        #pragma omp single 
        {
            promA /= S;
            promB /= S;

            factor = (maxA * maxB - minA * minB) / (promA * promB);
        }   

        // Multiplicación excalar factor x R
        #pragma omp for schedule(static)
        for (int i = 0; i < S; i++) {
            matrizR[i] *= factor;
        }

        // Calcula CxD y realiza la suma con R
        #pragma omp for nowait schedule(static)
        for (int i = 0; i < N; i += BS)
        {
            for (int j = 0; j < N; j += BS)
            {
                for (int k = 0; k < N; k += BS)
                {
                    blkmul(&matrizC[i*N + k], &matrizD[j*N + k], &matrizR[i*N + j], BS);
                }
            }
        }

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