#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <mpi.h>

/* Parsea los argumentos de entrada */
int parse_args(int argc, char **argv, int *N, int *B);

/* Inicializa las matrices con valores fijos para pruebas uniformes */
void init_matrices(double *a, double *b, double *c, double *d, double *r, int n);

/* Incializa una matriz con el valor val en todas sus celdas */
void init_matrix(double *m, int n, double val, bool transpose);

/* Realiza la multiplicacion de un bloque en multiplicación por bloques */
void blkmul(double *ablk, double *bblk, double *cblk, int N, int bs);

/* Realiza la multiplicacion de matrices por bloques*/
void matrix_multiplication(double *a, double *b, double *c, int N, int size, int bs);

/* Realiza una multiplicación escalar en una matriz */
void matrix_scalar_multiplication(double *m, int n, int size, double factor);

/* Validación de resultados finales */
void verify_results(double *c, int n);

/* Imprime información de ejecución */
void print_execution_data(int n, int bs, int num_procs, double commTimes, double *maxCommTimes, double *minCommTimes);

/* Obtiene el minimo, maximo y promedio de una matriz */
double metrics(double *matriz, int N, double *max, double *min, double *prom);

#define COORDINATOR 0

int main(int argc, char* argv[]) {
    int n, bs, numProcs, rank;
    int start_time, end_time;
    int process_lenght, process_rows, matrix_lenght, offset;
    double *a, *b, *c, *d, *r;
    double maxA, minA, promA, maxB, minB, promB;
    double commTimes[6], maxCommTimes[6], minCommTimes[6];
    MPI_Status status;

    parse_args(argc, argv, &n, &bs);
    matrix_lenght = n * n;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (n % numProcs != 0) {
	    printf("El largo de la matriz debe ser multiplo del numero de procesos.\n");
	    exit(1);
    }

    // calcular porcion de cada proceso
    process_rows = n / numProcs;
    process_lenght = process_rows * n;

    // Reservar memoria
    if (rank == COORDINATOR) {
	    a = (double*) malloc(sizeof(double)*matrix_lenght);
	    c = (double*) malloc(sizeof(double)*matrix_lenght);
        r = (double*) malloc(sizeof(double)*matrix_lenght);
    }
    else  {
	    a = (double*) malloc(sizeof(double)*process_lenght);
	    c = (double*) malloc(sizeof(double)*process_lenght);
        r = (double*) malloc(sizeof(double)*process_lenght);
    }

    // Para la multiplicación la matriz factor de la derecha se utiliza en su totalidad
    b = (double*) malloc(sizeof(double)*matrix_lenght);
    d = (double*) malloc(sizeof(double)*matrix_lenght);

    // inicializar datos
    if (rank == COORDINATOR) {
        init_matrices(a, b, c, d, r, n);
    }

    // Espera para que todos los procesos estén listos para medir los tiempos
    MPI_Barrier(MPI_COMM_WORLD);

    commTimes[0] = MPI_Wtime();

    MPI_Scatter(a, process_lenght, MPI_DOUBLE, a, process_lenght, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);
    MPI_Scatter(c, process_lenght, MPI_DOUBLE, c, process_lenght, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);

    MPI_Bcast(b, matrix_lenght, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);
    MPI_Bcast(d, matrix_lenght, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);

    commTimes[1] = MPI_Wtime();

    metrics(a, process_lenght, &maxA, &minA, &promA);
    promA = promA / matrix_lenght; // (a+b+c+...+n) / x = (a/x)+(b/x)+(c/x)+...+(n/x)
    // Como B se recibe completamente, cada proceso debe recorrer una porción diferente
    offset = process_rows * rank;
    metrics(b+offset, process_lenght, &maxB, &minB, &promB);
    promB = promB / matrix_lenght;

    commTimes[2] = MPI_Wtime();

    MPI_Allreduce(&maxA, &maxA, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&minA, &minA, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&promA, &promA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&maxB, &maxB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&minB, &minB, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&promB, &promB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    commTimes[3] = MPI_Wtime();

    double factor = (maxA * maxB - minA * minB) / (promA * promB);

    matrix_multiplication(a, b, r, n, process_rows, bs);
    matrix_scalar_multiplication(r, n, process_lenght, factor);
    matrix_multiplication(c, d, r, n, process_rows, bs);

    commTimes[4] = MPI_Wtime();

    MPI_Gather(r, process_lenght, MPI_DOUBLE, r, process_lenght, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);

    commTimes[5] = MPI_Wtime();

    MPI_Reduce(commTimes, minCommTimes, 6, MPI_DOUBLE, MPI_MIN, COORDINATOR, MPI_COMM_WORLD);
    MPI_Reduce(commTimes, maxCommTimes, 6, MPI_DOUBLE, MPI_MAX, COORDINATOR, MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank==COORDINATOR) {
        verify_results(r, n);
        print_execution_data(n, bs, numProcs, commTimes, maxCommTimes, minCommTimes);
    }

    free(a);
    free(b);
    free(c);
    free(d);
    free(r);

    return 0;
}


void init_matrices(double *a, double *b, double *c, double *d, double *r, int n)
{
    init_matrix(a, n, 1, false);
    init_matrix(c, n, 1, false);
    init_matrix(b, n, 1, true);
    init_matrix(d, n, 1, true);
    init_matrix(r, n, 0, false);
    /*
    Con estos valores la multiplicación de cada celda es N, luego:
        (maxA * maxB - minA * minB) / (promA * promB) = (3 * 1 - (-1) * 1) / (1 * 1) = 4
    Entonces el valor final del cálculo de R es N*5 en cada celda
    */
    a[0] = 3.0; a[1] = -1.0;
}


void matrix_scalar_multiplication(double *m, int n, int size, double factor) {
    for(int i = 0; i < size; i++) {
        m[i] *= factor;
    }
}


void init_matrix(double *m, int n, double val, bool transpose) {
    int i, j, row;

    if (transpose) {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                m[j*n+i] = val;
            }
        }
    }
    else {
        for (i = 0; i < n; i++)
        {
            row = i * n;
            for (j = 0; j < n; j++)
            {
                m[row+j] = val;
            }
        }
    }
}

double metrics(double *matriz, int n, double *max, double *min, double *prom) {
    *max = matriz[0];
    *min = matriz[0];
    *prom = 0.0;
    for (int i = 0; i < n; i++) {
        if (matriz[i] > *max) {
        *max = matriz[i];
        }
        if (matriz[i] < *min) {
        *min = matriz[i];
        }
        *prom += matriz[i];
    }
}


void blkmul(double *ablk, double *bblk, double *cblk, int N, int bs)
{
    int i, j, k, row, col, sum; 

    for (i = 0; i < bs; i++)
    {
        row = i * N;
        for (j = 0; j < bs; j++)
        {
            col = j * N; sum = 0;
            for  (k = 0; k < bs; k++)
            {
                sum += ablk[row + k] * bblk[col + k];
            }
            cblk[row + j]+= sum;
        }
    }
}

void matrix_multiplication(double *a, double *b, double *c, int N, int size, int bs)
{
    int row, col;
    for (int i = 0; i < size; i += bs)
    {
        row = i * N;
        for (int j = 0; j < N; j += bs)
        {
            col = j * N;
            for  (int k = 0; k < N; k += bs)
            {
                blkmul(&a[row + k], &b[col + k], &c[row + j], N, bs);
            }
        }
    }
}

void verify_results(double *c, int n) {
    int i, j, row, res = n*5;
    bool check = true;
    for (i = 0; i < n; i++) {
        row = i * n;
        for (j = 0; j < n; j++) {
            check = check && (c[row+j] == res);
        }
    }
    if (check) {
        printf("Multiplicacion de matrices resultado correcto\n");
    } else {
        printf("Multiplicacion de matrices resultado erroneo\n");
    }
}

void print_execution_data(int n, int bs, int num_procs, double *commTimes, double *maxCommTimes, double *minCommTimes) {
    
    // Como los clocks pueden no estar sincronizacos, se toma como total
    // el tiempo entre el primer y último tiempo de comunicación tomado por el coordinador
    // ya por al ser la última instrucción un Gather el coordinador es el último en terminar
    double totalTime = commTimes[5] - commTimes[0];

    double commTime = (maxCommTimes[1] - minCommTimes[0])
                    + (maxCommTimes[3] - minCommTimes[2])
                    + (maxCommTimes[5] - minCommTimes[4]);
                
    printf("Tiempo total: %f\n",  totalTime);
    printf("Tiempo de comunicación: %f\n", commTime);
    printf("Matriz de %dx%d\n", n, n);
    printf("Largo de bloque %d\n", bs);
    printf("Cantidad de procesos %d\n", num_procs);  
}


#define DEFAULT_OPT_ARG_N 512
#define DEFAULT_OPT_ARG_B 32

#define OPT_STRING "n:b:h"
#define HELP_MESSAGE \
    "Este programa realiza ((maxA * maxB - minA * minB) / promA * promB) x [A x B] + [C x D]:\n" \
    "-n, --size <number+>: define el largo de la matriz. Por defecto es 512\n" \
    "-b, --block-size <number+>: define el largo del bloque para el producto por bloques. Por defecto es 32 \n" \
    "-h, --help: muestra este mensaje de ayuda\n"

int parse_args(int argc, char **argv, int *N, int *B) {
    // Configuration OPT
    extern char *optarg;
    extern int opterr;
    opterr = 0; // Suppress getopt errors

    static struct option long_options[] = {
        {"size", required_argument, 0, 'n'},
        {"block-size", required_argument, 0, 'b'},
        {"help", no_argument, 0, 'h'},
    };

    // OPT Proccessing
    *N = DEFAULT_OPT_ARG_N;
    *B = DEFAULT_OPT_ARG_B;

    int opt, arg;
    while ((opt = getopt_long(argc, argv, OPT_STRING, long_options, NULL)) != -1) {
        switch (opt) {
            case 'n':
                arg = atoi(optarg);
                *N = (arg > 0) ? arg : DEFAULT_OPT_ARG_N;
                break;

            case 'b':
                arg = atoi(optarg);
                *B = (arg > 0) ? arg : DEFAULT_OPT_ARG_B;
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