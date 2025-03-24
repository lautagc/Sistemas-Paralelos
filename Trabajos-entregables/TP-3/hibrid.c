#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <omp.h>
#include <mpi.h>

/* Parsea los argumentos de entrada */
int parse_args(int argc, char **argv, int *N, int *B, int *T);

/* Inicializa las matrices con valores fijos para pruebas uniformes */
void init_matrices(double *a, double *b, double *c, double *d, double *r, int n);

/* Incializa una matriz con el valor val en todas sus celdas */
void init_matrix(double *m, int n, double val, bool transpose);

/* Realiza la multiplicacion de un bloque en multiplicación por bloques */
void blkmul(double *ablk, double *bblk, double *cblk, int N, int bs);

void scalar_blkmul(double *ablk, double *bblk, double *cblk, double factor, int N, int bs);

/* Validación de resultados finales */
void verify_results(double *c, int n);

/* Imprime información de ejecución */
void print_execution_data(int n, int bs, int t, int num_procs, double *commTimes, double commPromTime);

#define COORDINATOR 0

int main(int argc, char* argv[]) {
	int n, bs, t, numProcs, rank, provided;
    int process_lenght, process_rows, matrix_lenght;
	double *a, *b, *c, *d, *r;
	double commTimes[6];
    MPI_Status status;

    parse_args(argc, argv, &n, &bs, &t) ;
    matrix_lenght = n * n;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided != MPI_THREAD_SERIALIZED) {
        printf("El nivel de soporte de MPI no es el requerido\n");
        exit(1);
    }

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

    int i, j, k, row, col;
    double factor;
    double *sb = b + process_rows * rank; //subset of b, offset = process_rows * rank
    double maxA = DBL_MIN, minA = DBL_MAX, promA = 0;
    double maxB = DBL_MIN, minB = DBL_MAX, promB = 0;
    
    #pragma omp parallel private(i, j, k) num_threads(t)
    { 
        // Calcula minimo, maximo y promedio de A
        #pragma omp for reduction(min: minA) reduction(max: maxA) reduction(+:promA) schedule(static)
        for (i = 0; i < process_lenght; i++) {
            if (a[i] > maxA) {
                maxA = a[i];
            }
            if (a[i] < minA) {
                minA = a[i];
            }
            promA += a[i];
        }

        // Calcula minimo, maximo y promedio de B
        #pragma omp for reduction(min: minB) reduction(max: maxB) reduction(+:promB) schedule(static)
        for (i = 0; i < process_lenght; i++) {
            if (sb[i] > maxB) {
                maxB = sb[i];
            }
            if (sb[i] < minB) {
                minB = sb[i];
            }
            promB += sb[i];
        }
        
        #pragma omp single
        {
            commTimes[2] = MPI_Wtime();
            
            MPI_Allreduce(&maxA, &maxA, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&minA, &minA, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&promA, &promA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(&maxB, &maxB, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&minB, &minB, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&promB, &promB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            commTimes[3] = MPI_Wtime();
        }

        // Calculo del factor para multiplicación escalar
        #pragma omp single
        {
            promA = promA / matrix_lenght;
            promB = promB / matrix_lenght;
            factor = (maxA * maxB - minA * minB) / (promA * promB);
        }

        // Multiplicación de AxB y multiplicación escalar
        #pragma omp for private(j, k, row, col) schedule(static) nowait
        for (int i = 0; i < process_rows; i += bs)
        {
            row = i * n;
            for (j = 0; j < n; j += bs)
            {
                col = j * n;
                for  (k = 0; k < n; k += bs)
                {
                    scalar_blkmul(&a[row + k], &b[col + k], &r[row + j], factor, n, bs);
                }
            }
        }

        // Multiplicación de CxD
        #pragma omp for private(j, k, row, col) schedule(static) nowait
        for (int i = 0; i < process_rows; i += bs)
        {
            row = i * n;
            for (j = 0; j < n; j += bs)
            {
                col = j * n;
                for  (k = 0; k < n; k += bs)
                {
                    blkmul(&c[row + k], &d[col + k], &r[row + j], n, bs);
                }
            }
        }
    }

	commTimes[4] = MPI_Wtime();

    MPI_Gather(r, process_lenght, MPI_DOUBLE, r, process_lenght, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);
	
    commTimes[5] = MPI_Wtime();

    // Medición de tiempos, debido a la asincronía de los clocks se toman los tiempos de comunicación
    // calculando un promedio entre las distancia de los diferentes tiempos tomados
    double localCommTime = (commTimes[1] - commTimes[0])
                        + (commTimes[3] - commTimes[2])
                        + (commTimes[5] - commTimes[4]);
    double globalCommTime;

    MPI_Reduce(&localCommTime, &globalCommTime, 1, MPI_DOUBLE, MPI_SUM, COORDINATOR, MPI_COMM_WORLD);

    if (rank == COORDINATOR) {
        globalCommTime = globalCommTime / numProcs;
    }

	MPI_Finalize();

	if (rank==COORDINATOR) {
        verify_results(r, n);
        print_execution_data(n, bs, t, numProcs, commTimes, globalCommTime);
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
    init_matrix(a, n, 1.0, false);
    init_matrix(c, n, 1.0, false);
    init_matrix(b, n, 1.0, true);
    init_matrix(d, n, 1.0, true);
    init_matrix(r, n, 0.0, false);
    /*
    Con estos valores la multiplicación de cada celda es N, luego:
        (maxA * maxB - minA * minB) / (promA * promB) = (3 * 1 - (-1) * 1) / (1 * 1) = 4
    Entonces el valor final del cálculo de R es N*5 en cada celda
    */
    a[0] = 3.0; a[1] = -1.0;
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

void scalar_blkmul(double *ablk, double *bblk, double *cblk, double factor, int N, int bs)
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
            cblk[row + j]+= sum * factor;
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

void print_execution_data(int n, int bs, int t, int num_procs, double *commTimes, double commPromTime) {
    // Como los clocks pueden no estar sincronizacos, se toma como total
    // el tiempo entre el primer y último tiempo de comunicación tomado por el coordinador
    // ya por al ser la última instrucción un Gather el coordinador es el último en terminar
    double totalTime = commTimes[5] - commTimes[0];

    double commTime = commPromTime;
                
    printf("Tiempo total: %f\n",  totalTime);
    printf("Tiempo de comunicación: %f\n", commTime);
    printf("Matriz de %dx%d\n", n, n);
    printf("Largo de bloque %d\n", bs);
    printf("Cantidad de procesos %d\n", num_procs);  
    printf("Cantidad de hilos %d\n", t);
}

#define DEFAULT_OPT_ARG_N 512
#define DEFAULT_OPT_ARG_B 32
#define DEFAULT_OPT_ARG_T 4

#define OPT_STRING "n:b:t:h"

#define HELP_MESSAGE \
    "Este programa realiza ((maxA * maxB - minA * minB) / promA * promB) x [A x B] + [C x D]:\n" \
    "-n, --size <number+>: define el largo de la matriz. Por defecto es 512\n" \
    "-b, --block-size <number+>: define el largo del bloque para el producto por bloques. Por defecto es 32 \n" \
    "-t, --threads <number+>: define la cantidad de hilos a utilizar. Por defecto es 4\n"

int parse_args(int argc, char **argv, int *N, int *B, int *T) {
    // Configuration OPT
    extern char *optarg;
    extern int opterr;
    opterr = 0; // Suppress getopt errors

    static struct option long_options[] = {
        {"size", required_argument, 0, 'n'},
        {"block-size", required_argument, 0, 'b'},
        {"threads", required_argument, 0, 't'},
        {"help", no_argument, 0, 'h'},
    };

    // OPT Processing
    *N = DEFAULT_OPT_ARG_N;
    *B = DEFAULT_OPT_ARG_B;
    *T = DEFAULT_OPT_ARG_T;

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

        case 't':
            arg = atoi(optarg);
            *T = (arg > 0) ? arg : DEFAULT_OPT_ARG_T;
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

    return 0;
}
