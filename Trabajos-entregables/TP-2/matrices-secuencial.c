#include <stdbool.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* Time in seconds from some point in the past */
double dwalltime();

/* Parsear argumentos */
int parse_args(int argc, char **argv, int *N, int *B);

/* Inicializa las matrices con valores fijos para pruebas uniformes */
void init_matrices(double *matrizA,  double *matrizB, double *matrizC,
    double *matrizD, double *matrizR);

/* Obtiene el minimo, máximo y promedio de una matriz */
double metrics(double *matriz, double *max, double *min, double *prom);

/* Realiza la multiplicación por bloques */
void blkmul(double *ablk, double *bblk, double *cblk, int bs);

/* Realiza el producto y suma de las matrices */
void producto_matriz(double *a, double *b, double *c, int bs);

/* Realiza la multipliación escalar de la matriz */
void producto_escalar(double *matriz, double factor);

/* Validación de los resultados finales */
bool verify_results(double *matrizR);

int N,  BS;

int main(int argc, char *argv[]) {

    // Parsear argumentos
    parse_args(argc, argv, &N, &BS);

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

    init_matrices(matrizA, matrizB, matrizC, matrizD, matrizR);

    time = dwalltime();

    // Calcular maximo, minimo y promedio
    metrics(matrizA, &maxA, &minA, &promA);
    metrics(matrizB, &maxB, &minB, &promB);

    // Calcular los productos en R
    producto_matriz(matrizA, matrizB, matrizR, BS);
    double factor = (maxA * maxB - minA * minB) / (promA * promB);
    producto_escalar(matrizR, factor);
    
    // Calcular el producto con CxD y realizar la suma final
    producto_matriz(matrizC, matrizD, matrizR, BS);

    // Información de ejecución
    printf("Tiempo: %f\n", dwalltime() - time);
    printf("Matriz de %dx%d\n", N, N);
    printf("Largo de bloque %d\n", BS);

    // Verificar resultados
    if (verify_results(matrizR)) {
        printf("Los resultados son correctos\n");
    }
    else {
        printf("Los resultados son incorrectos\n");
    }
    
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

void producto_matriz(double *a, double *b, double *c, int bs)
{
    int i, j, k, row, col;

    for (i = 0; i < N; i += bs)
    {
        row = i*N;
        for (j = 0; j < N; j += bs)
        {
            col = j*N;
            for  (k = 0; k < N; k += bs)
            {
                blkmul(&a[row + k], &b[col + k], &c[row + j], bs);
            }
        }
    }
}

void producto_escalar(double *matriz, double factor)
{
    for(int i = 0; i < N*N; i++) {
        matriz[i] *= factor;
    }
}

int parse_args(int argc, char **argv, int *N, int *B) {
    // Configuration OPT
    extern char *optarg;
    extern int  opterr;
    opterr = 0; //Supress getopt errors

    const char *OPT_STRING = "n:b:h";
    const int DEFAULT_OPT_ARG_N = 512;
    const int DEFAULT_OPT_ARG_B = 32;
    const char *HELP_MESSAGE = 
        "Este programa realiza ((maxA * maxB - minA * minB) / promA * promB) x [A x B] + [C x D]:\n"
        "-n, --size <number+>: define el largo de la matriz. Por defecto es 512\n"
        "-b, --block-size <number+>: define el largo del bloque para el producto por bloques. Por defecto es 32 \n"
    ;

    static struct option long_options[] = {
        {"size", required_argument, 0, 'n'},
        {"block-size", required_argument, 0, 'b'},
        {"help", no_argument, 0, 'h'},
    };

    // OPT Proccessing
    *N = DEFAULT_OPT_ARG_N;
    *B = DEFAULT_OPT_ARG_B;

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