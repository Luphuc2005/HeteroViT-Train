#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <stddef.h>
#include <unistd.h>

/*
 * Interceptor for OpenMPI 4.0.3 Allreduce internal collective functions.
 * Captures which exact internal collective algorithm is called by ompi_coll_tuned.
 */

// 1. Segmented Ring
int ompi_coll_base_allreduce_intra_ring_segmented(void *sbuf, void *rbuf, int count,
                                                   void *dtype, void *op,
                                                   void *comm, void *module, size_t segsize) {
    static int (*real_fn)(void*, void*, int, void*, void*, void*, void*, size_t) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "ompi_coll_base_allreduce_intra_ring_segmented");
    fprintf(stderr, "[INTERCEPT EVIDENCE][PID %d] >>> ALGORITHM DETECTED: ompi_coll_base_allreduce_intra_ring_segmented | count=%d | segsize=%zu bytes\n",
            getpid(), count, segsize);
    return real_fn(sbuf, rbuf, count, dtype, op, comm, module, segsize);
}

// 2. Unsegmented Ring
int ompi_coll_base_allreduce_intra_ring(void *sbuf, void *rbuf, int count,
                                        void *dtype, void *op,
                                        void *comm, void *module) {
    static int (*real_fn)(void*, void*, int, void*, void*, void*, void*) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "ompi_coll_base_allreduce_intra_ring");
    fprintf(stderr, "[INTERCEPT EVIDENCE][PID %d] >>> ALGORITHM DETECTED: ompi_coll_base_allreduce_intra_ring | count=%d\n",
            getpid(), count);
    return real_fn(sbuf, rbuf, count, dtype, op, comm, module);
}

// 3. Recursive Doubling
int ompi_coll_base_allreduce_intra_recursivedoubling(void *sbuf, void *rbuf, int count,
                                                     void *dtype, void *op,
                                                     void *comm, void *module) {
    static int (*real_fn)(void*, void*, int, void*, void*, void*, void*) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "ompi_coll_base_allreduce_intra_recursivedoubling");
    fprintf(stderr, "[INTERCEPT EVIDENCE][PID %d] >>> ALGORITHM DETECTED: ompi_coll_base_allreduce_intra_recursivedoubling | count=%d\n",
            getpid(), count);
    return real_fn(sbuf, rbuf, count, dtype, op, comm, module);
}

// 4. Reduce-Scatter + Allgather (Rabenseifner)
int ompi_coll_base_allreduce_intra_redscat_allgather(void *sbuf, void *rbuf, int count,
                                                     void *dtype, void *op,
                                                     void *comm, void *module, size_t segsize) {
    static int (*real_fn)(void*, void*, int, void*, void*, void*, void*, size_t) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "ompi_coll_base_allreduce_intra_redscat_allgather");
    fprintf(stderr, "[INTERCEPT EVIDENCE][PID %d] >>> ALGORITHM DETECTED: ompi_coll_base_allreduce_intra_redscat_allgather | count=%d\n",
            getpid(), count);
    return real_fn(sbuf, rbuf, count, dtype, op, comm, module, segsize);
}

// 5. Nonoverlapping (Reduce + Bcast)
int ompi_coll_base_allreduce_intra_nonoverlapping(void *sbuf, void *rbuf, int count,
                                                  void *dtype, void *op,
                                                  void *comm, void *module) {
    static int (*real_fn)(void*, void*, int, void*, void*, void*, void*) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "ompi_coll_base_allreduce_intra_nonoverlapping");
    fprintf(stderr, "[INTERCEPT EVIDENCE][PID %d] >>> ALGORITHM DETECTED: ompi_coll_base_allreduce_intra_nonoverlapping | count=%d\n",
            getpid(), count);
    return real_fn(sbuf, rbuf, count, dtype, op, comm, module);
}

// 6. Basic Linear
int ompi_coll_base_allreduce_intra_basic_linear(void *sbuf, void *rbuf, int count,
                                                void *dtype, void *op,
                                                void *comm, void *module) {
    static int (*real_fn)(void*, void*, int, void*, void*, void*, void*) = NULL;
    if (!real_fn) real_fn = dlsym(RTLD_NEXT, "ompi_coll_base_allreduce_intra_basic_linear");
    fprintf(stderr, "[INTERCEPT EVIDENCE][PID %d] >>> ALGORITHM DETECTED: ompi_coll_base_allreduce_intra_basic_linear | count=%d\n",
            getpid(), count);
    return real_fn(sbuf, rbuf, count, dtype, op, comm, module);
}

