/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/spmm.h
 * \brief SPMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SPMM_H_
#define DGL_ARRAY_CPU_SPMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <limits>
#include <algorithm>

namespace dgl {
namespace aten {
namespace cpu {

/*!
 * \brief CPU kernel of SpMM on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */

#if 1
// // -----------------------------------------------------------------------------------
// // ------------------------------ Optimized Sparse MM3 -------------------------------
// // -----------------------------------------------------------------------------------
template <typename IdType, typename DType, typename Op>
// void sparse_mm3(
  void SpMMSumCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out) {

  const IdType* IndPtr = csr.indptr.Ptr<IdType>();
  const IdType* Indices = csr.indices.Ptr<IdType>();
  // const IdType* edges = csr.data.Ptr<IdType>();
  
  DType* C = out.Ptr<DType>();
  DType* B = ufeat.Ptr<DType>();

  if(sizeof(DType) == 8)
    std::cout << "sizeof DType" << sizeof(DType) << " Sizeof IdType " << sizeof(IdType) << std::endl;


  #define M_BLOCK_SIZE 1024
  #define K_BLOCK_SIZE 4096
  #define K_BLOCK_MASK (K_BLOCK_SIZE - 1)
  #define N_BLOCK_SIZE 640
  #define SORT 0
    const int M = csr.num_rows;
    const int N = bcast.out_len; //csr.N;
    const int K = csr.num_cols;
    int nthreads = omp_get_max_threads();

    int32_t num_M_blocks = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;
    int32_t num_K_blocks = (K + K_BLOCK_SIZE - 1) / K_BLOCK_SIZE;

    csrm block_csr_array[num_M_blocks * num_K_blocks];
    //int *cur_col_id = (int *)_mm_malloc(2 * M_BLOCK_SIZE * sizeof(int), 64);

    uint64_t startTick, endTick;
    startTick = __rdtsc();
    #pragma omp parallel
    {
        int *my_cur_col_id = (int *)_mm_malloc(2 * M_BLOCK_SIZE * sizeof(int), 64);
        uint64_t tst = __rdtsc();
         
        int tid = omp_get_thread_num();     
        #pragma omp for
        for(int m = 0; m < num_M_blocks; m++)
        {
            int32_t M_start = m * M_BLOCK_SIZE;
            int32_t M_end = (m + 1) * M_BLOCK_SIZE;
            if(M_end > M) M_end = M;
            int nnz = IndPtr[M_end] - IndPtr[M_start];
            int32_t cur_indices_id = 0;
            int32_t *indices = (int32_t *)_mm_malloc(nnz * sizeof(int32_t), 64);

            for(int i = M_start; i < M_end; i++)
            {
                my_cur_col_id[(i - M_start) * 2] = IndPtr[i];
                my_cur_col_id[(i - M_start) * 2 + 1] = IndPtr[i + 1];
            }
            for(int k = 0; k < num_K_blocks; k++)
            {
                int32_t K_start = k * K_BLOCK_SIZE;
                int32_t K_end = (k + 1) * K_BLOCK_SIZE;
                if(K_end > K) K_end = K;
                csrm cur_csr;
                cur_csr.M = M_end - M_start;
                cur_csr.K = K_end - K_start;
                cur_csr.N = N;
                // Create csr_ij
                int32_t *indptr = (int32_t *)_mm_malloc((cur_csr.M + 1) * sizeof(int32_t), 64);
                cur_csr.indptr = indptr;
                cur_csr.indices = indices + cur_indices_id;
                cur_csr.values = NULL;
                int cur_nnz = 0;
                for(int i = M_start; i < M_end; i++)
                {
                    const int row_start = my_cur_col_id[(i - M_start) * 2];
                    const int row_end   = my_cur_col_id[(i - M_start) * 2 + 1];
                    indptr[i - M_start] = cur_nnz;
                    int eid;
                    for(eid = row_start; eid < row_end; eid++)
                    {
                        const int dst = Indices[eid];
                        if(dst >= K_end)
                        {
                            break;
                        }
                        if(cur_indices_id + cur_nnz >= nnz)
                        {
                            printf("Error! cur_indices_id + cur_nnz = %d, nnz = %d\n", cur_indices_id + cur_nnz, nnz);
                            exit(0);
                        }
                        indices[cur_indices_id + cur_nnz] = dst;
                        cur_nnz++;
                    }
                    my_cur_col_id[(i - M_start) * 2] = eid;
                }
                indptr[cur_csr.M] = cur_nnz;
                cur_indices_id += cur_nnz;
                block_csr_array[m * num_K_blocks + k] = cur_csr;

            }
            if(nnz != cur_indices_id)
            {
                printf("cur_indices_id = %d, expected = %d\n", cur_indices_id, nnz);
                exit(0);
            }
        }
        _mm_free(my_cur_col_id);
        uint64_t tend = __rdtsc();
        // printf("%d] %lu\n", tid, tend - tst);
    }
    endTick = __rdtsc();
    // printf("stage 1: %lu\n", endTick - startTick);
    // int nnz_ = static_cast<int32_t*>(csr.indptr)[M];
    int nnz_ = static_cast<const IdType*>(IndPtr)[M];

    #if VER
    fprintf(stderr, "nthreads: %d, M: %d, K: %d, N: %d, nzz: %d\n",
            nthreads, M, K, N, nnz_);
    #endif
    
    // #if FILEIO
    // static int cnt = 0;
    // if (N > 600) {
    //     cnt ++;
    //     FILE *fp = fopen("csr.txt", "a");
    //     fwrite(&M, sizeof(int32_t), 1, fp);
    //     fwrite(&K, sizeof(int32_t), 1, fp);
    //     fwrite(&N, sizeof(int32_t), 1, fp);
    //     fwrite(static_cast<int32_t*>(csr.indptr->data), sizeof(int32_t), M+1, fp);
    //     fwrite(static_cast<int32_t*>(csr.indices->data), sizeof(int32_t), nnz_, fp);
    //     fclose(fp);
    // }
    // if (cnt == 300)
    //     exit(0);
    // #endif
   
#define PFD 160
    startTick = __rdtsc();
    int32_t N_block_start = 0;
    int32_t N_block_end = N;
    int rem = (N_block_end - N_block_start) & 0xf;
    __mmask16 mask = (1 << rem) - 1;
    __m512 zero512 = _mm512_setzero_ps();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();     
        uint64_t tst = __rdtsc();
        for(int32_t k = 0; k < num_K_blocks; k++)
        {
#pragma omp for schedule(dynamic)
            for(int32_t m = 0; m < num_M_blocks; m++)
            {
                //printf("m = %d\n", m);
                csrm cur_csr = block_csr_array[m * num_K_blocks + k];

                int32_t cur_M = cur_csr.M;
                int32_t cur_K = cur_csr.K;
                int32_t cur_N = cur_csr.N;

                int32_t M_start = m * M_BLOCK_SIZE;
                for(int i = 0; i < cur_M; i++)
                {
                    const int row_start = cur_csr.indptr[i];
                    const int row_end   = cur_csr.indptr[i + 1];
                    int32_t src = i + M_start;

                    int32_t eid;
                    for(eid = row_start; eid < (row_end - 4); eid+=4)
                    {
                        int j;
                        DType *Bptr0 = &B[cur_csr.indices[eid] * N + N_block_start];
                        DType *Bptr1 = &B[cur_csr.indices[eid + 1] * N + N_block_start];
                        DType *Bptr2 = &B[cur_csr.indices[eid + 2] * N + N_block_start];
                        DType *Bptr3 = &B[cur_csr.indices[eid + 3] * N + N_block_start];
                        DType *B_next_ptr0 = &B[cur_csr.indices[eid + 4] * N + N_block_start];
                        DType *B_next_ptr1 = &B[cur_csr.indices[eid + 5] * N + N_block_start];
                        DType *B_next_ptr2 = &B[cur_csr.indices[eid + 6] * N + N_block_start];
                        DType *B_next_ptr3 = &B[cur_csr.indices[eid + 7] * N + N_block_start];
                        DType *Cptr = &C[src * N + N_block_start];
#pragma unroll(16)
                        for(j = N_block_start; j < N_block_end - PFD; j += 16)
                        {
                            _mm_prefetch((const char *)(Bptr0 + PFD), _MM_HINT_T0);
                            _mm_prefetch((const char *)(Bptr1 + PFD), _MM_HINT_T0);
                            _mm_prefetch((const char *)(Bptr2 + PFD), _MM_HINT_T0);
                            _mm_prefetch((const char *)(Bptr3 + PFD), _MM_HINT_T0);
                            //B_next_ptr0 += 16;
                            //B_next_ptr1 += 16;
                            //B_next_ptr2 += 16;
                            //B_next_ptr3 += 16;
                            __m512 c512 = _mm512_loadu_ps(Cptr);
                            Cptr += 16;
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr0), c512);
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr1), c512);
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr2), c512);
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr3), c512);
                            Bptr0 += 16;
                            Bptr1 += 16;
                            Bptr2 += 16;
                            Bptr3 += 16;
                            _mm512_storeu_ps(&C[src * N + j], c512);
                        }
#pragma unroll(16)
                        for(; j < N_block_end - 15; j += 16)
                        {
                            _mm_prefetch((const char *)(B_next_ptr0), _MM_HINT_T0);
                            _mm_prefetch((const char *)(B_next_ptr1), _MM_HINT_T0);
                            _mm_prefetch((const char *)(B_next_ptr2), _MM_HINT_T0);
                            _mm_prefetch((const char *)(B_next_ptr3), _MM_HINT_T0);
                            B_next_ptr0 += 16;
                            B_next_ptr1 += 16;
                            B_next_ptr2 += 16;
                            B_next_ptr3 += 16;
                            __m512 c512 = _mm512_loadu_ps(Cptr);
                            Cptr += 16;
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr0), c512);
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr1), c512);
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr2), c512);
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr3), c512);
                            Bptr0 += 16;
                            Bptr1 += 16;
                            Bptr2 += 16;
                            Bptr3 += 16;
                            _mm512_storeu_ps(&C[src * N + j], c512);
                        }
                        __m512 c512 = _mm512_mask_loadu_ps(zero512, mask, &C[src * N + j]);
                        c512 = _mm512_add_ps(_mm512_mask_loadu_ps(zero512, mask, Bptr0), c512);
                        c512 = _mm512_add_ps(_mm512_mask_loadu_ps(zero512, mask, Bptr1), c512);
                        c512 = _mm512_add_ps(_mm512_mask_loadu_ps(zero512, mask, Bptr2), c512);
                        c512 = _mm512_add_ps(_mm512_mask_loadu_ps(zero512, mask, Bptr3), c512);
                        _mm512_mask_storeu_ps(&C[src * N + j], mask, c512);
                    }
                    for(; eid < (row_end - 1); eid++)
                    {
                        int32_t dst = cur_csr.indices[eid];
                        int32_t dst_next = cur_csr.indices[eid + 1];
                        int j;
                        DType *Bptr = &B[dst * N + N_block_start];
                        DType *B_next_ptr = &B[dst_next * N + N_block_start];
                        DType *Cptr = &C[src * N + N_block_start];
#pragma unroll(16)
                        for(j = N_block_start; j < N_block_end - 15; j += 16)
                        {
                            _mm_prefetch((const char *)(B_next_ptr), _MM_HINT_T0);
                            B_next_ptr += 16;
                            __m512 c512 = _mm512_loadu_ps(Cptr);
                            Cptr += 16;
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr), c512);
                            Bptr += 16;
                            _mm512_storeu_ps(&C[src * N + j], c512);
                            // if (N < 602)
                            //     std::cout << " Check eid " << N << std::endl;
                        }
                        // if (N < 602)
                        // {    std::cout << " Start loop --- outer " << N << std::endl;
                        //     std::cout << " Start Block " << N_block_start << std::endl;
                        //     std::cout << " End Block " << N_block_end << std::endl;
                        // }
                        __m512 c512 = _mm512_mask_loadu_ps(zero512, mask, &C[src * N + j]);
                        c512 = _mm512_add_ps(_mm512_mask_loadu_ps(zero512, mask, Bptr), c512);
                        _mm512_mask_storeu_ps(&C[src * N + j], mask, c512);
                        
                    }
                    for(; eid < row_end; eid++)
                    {
                        int32_t src_next = src + 1;
                        int32_t dst = cur_csr.indices[eid];
                        int32_t dst_next = cur_csr.indices[eid + 1];
                        DType *Bptr = &B[dst * N + N_block_start];
                        DType *B_next_ptr = &B[dst_next * N + N_block_start];
                        DType *Cptr = &C[src * N + N_block_start];
                        DType *C_next_ptr = &C[src_next * N + N_block_start];
                        int j;
#pragma unroll(16)
                        for(j = N_block_start; j < N_block_end - 15; j += 16)
                        {
                            _mm_prefetch((const char *)(C_next_ptr), _MM_HINT_T0);
                            C_next_ptr += 16;
                            _mm_prefetch((const char *)(B_next_ptr), _MM_HINT_T0);
                            B_next_ptr += 16;
                            __m512 c512 = _mm512_loadu_ps(Cptr);
                            Cptr += 16;
                            c512 = _mm512_add_ps(_mm512_loadu_ps(Bptr), c512);
                            Bptr += 16;
                            _mm512_storeu_ps(&C[src * N + j], c512);
                        }
                        __m512 c512 = _mm512_mask_loadu_ps(zero512, mask, &C[src * N + j]);
                        c512 = _mm512_add_ps(_mm512_mask_loadu_ps(zero512, mask, Bptr), c512);
                        _mm512_mask_storeu_ps(&C[src * N + j], mask, c512);
                    }
                }
            }
        }
        uint64_t tend = __rdtsc();
        // printf("%d] %lu\n", tid, tend - tst);
    }
    endTick = __rdtsc();
    // printf("stage2 ticks = %ld\n", endTick - startTick);

    for(int m = 0; m < num_M_blocks; m++)
    {
        for(int k = 0; k < num_K_blocks; k++)
        {
            _mm_free(block_csr_array[m * num_K_blocks + k].indptr);
        }
        _mm_free(block_csr_array[m * num_K_blocks].indices);
    }
    #undef K_BLOCK_SIZE
    #undef K_BLOCK_MASK
    #undef N_BLOCK_SIZE
    #undef SORT    
}

#else
// // -------------------------------------------------------------------------------------------------
// // --------------------------------------- Default SpMMSumCsr --------------------------------------
// // -------------------------------------------------------------------------------------------------
template <typename IdType, typename DType, typename Op>
void SpMMSumCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  const DType* X = ufeat.Ptr<DType>();
  const DType* W = efeat.Ptr<DType>();
  int64_t dim = bcast.out_len,
          lhs_dim = bcast.lhs_len,
          rhs_dim = bcast.rhs_len;
  DType* O = out.Ptr<DType>();
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType *out_off = O + rid * dim;
    std::fill(out_off, out_off + dim, 0);
    for (IdType j = row_start; j < row_end; ++j) {
      const IdType cid = indices[j];
      const IdType eid = has_idx ? edges[j] : j;
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
        const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
        const DType *lhs_off =
            Op::use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
        const DType *rhs_off =
            Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
        out_off[k] += Op::Call(lhs_off, rhs_off);
      }
    }
  }
}
#endif
/*!
 * \brief CPU kernel of SpMM on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes. To avoid possible data hazard,
 *       we use atomic operators in the reduction phase.
 */
template <typename IdType, typename DType, typename Op>
void SpMMSumCoo(
    const BcastOff& bcast,
    const COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* edges = coo.data.Ptr<IdType>();
  const DType* X = ufeat.Ptr<DType>();
  const DType* W = efeat.Ptr<DType>();
  int64_t dim = bcast.out_len,
          lhs_dim = bcast.lhs_len,
          rhs_dim = bcast.rhs_len;
  DType* O = out.Ptr<DType>();
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
  memset(O, 0, out.GetSize());
  // spmm
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx? edges[i] : i;
    DType* out_off = O + cid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off = Op::use_lhs? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + rhs_add : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
      if (val != 0) {
#pragma omp atomic
        out_off[k] += val;
      }
    }
  }
}

/*!
 * \brief CPU kernel of SpMM-Min/Max on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices 
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices 
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \note It uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 * \note The result will contain infinity for zero-degree nodes.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len,
                lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    IdType* argx_off = argX + rid * dim;
    IdType* argw_off = argW + rid * dim;
    std::fill(out_off, out_off + dim, Cmp::zero);
    if (Op::use_lhs)
      std::fill(argx_off, argx_off + dim, 0);
    if (Op::use_rhs)
      std::fill(argw_off, argw_off + dim, 0);
    for (IdType j = row_start; j < row_end; ++j) {
      const IdType cid = indices[j];
      const IdType eid = has_idx? edges[j] : j;
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
        const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
        const DType* lhs_off = Op::use_lhs? X + cid * lhs_dim + lhs_add : nullptr;
        const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + rhs_add : nullptr;
        const DType val = Op::Call(lhs_off, rhs_off);
        if (Cmp::Call(out_off[k], val)) {
          out_off[k] = val;
          if (Op::use_lhs)
            argx_off[k] = cid;
          if (Op::use_rhs)
            argw_off[k] = eid;
        }
      }
    }
  }
}

/*!
 * \brief CPU kernel of SpMM-Min/Max on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices 
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices 
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes. To avoid possible data hazard,
 *       we use atomic operators in the reduction phase.
 * \note The result will contain infinity for zero-degree nodes.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCoo(
    const BcastOff& bcast,
    const COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges = has_idx? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len,
                lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
  std::fill(O, O + out.NumElements(), Cmp::zero);
  // spmm
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx? edges[i] : i;
    DType* out_off = O + cid * dim;
    IdType* argx_off = Op::use_lhs? argX + cid * dim : nullptr;
    IdType* argw_off = Op::use_rhs? argW + cid * dim : nullptr;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off = Op::use_lhs? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + rhs_add : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
#pragma omp critical
      if (Cmp::Call(out_off[k], val)) {
        out_off[k] = val;
        if (Op::use_lhs)
          argx_off[k] = rid;
        if (Op::use_rhs)
          argw_off[k] = eid;
      }
    }
  }
}

namespace op {

//////////////////////////////// binary operators on CPU ////////////////////////////////
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off + *rhs_off;
  }
};
template <typename DType> constexpr bool Add<DType>::use_lhs;
template <typename DType> constexpr bool Add<DType>::use_rhs;

template <typename DType>
struct Sub {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off - *rhs_off;
  }
};
template <typename DType> constexpr bool Sub<DType>::use_lhs;
template <typename DType> constexpr bool Sub<DType>::use_rhs;

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off * *rhs_off;
  }
};
template <typename DType> constexpr bool Mul<DType>::use_lhs;
template <typename DType> constexpr bool Mul<DType>::use_rhs;

template <typename DType>
struct Div {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off / *rhs_off;
  }
};
template <typename DType> constexpr bool Div<DType>::use_lhs;
template <typename DType> constexpr bool Div<DType>::use_rhs;

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(const DType* lhs_off, const DType* ) {
    return *lhs_off;
  }
};
template <typename DType> constexpr bool CopyLhs<DType>::use_lhs;
template <typename DType> constexpr bool CopyLhs<DType>::use_rhs;

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* , const DType* rhs_off) {
    return *rhs_off;
  }
};
template <typename DType> constexpr bool CopyRhs<DType>::use_lhs;
template <typename DType> constexpr bool CopyRhs<DType>::use_rhs;

//////////////////////////////// Reduce operators on CPU ////////////////////////////////
template <typename DType>
struct Max {
  static constexpr DType zero = -std::numeric_limits<DType>::infinity();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) {
    return accum < val;
  }
};
template <typename DType> constexpr DType Max<DType>::zero;

template <typename DType>
struct Min {
  static constexpr DType zero = std::numeric_limits<DType>::infinity();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) {
    return accum > val;
  }
};
template <typename DType> constexpr DType Min<DType>::zero;

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef dgl::aten::cpu::op::Add<DType> Op;                    \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "sub") {                                     \
      typedef dgl::aten::cpu::op::Sub<DType> Op;                    \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef dgl::aten::cpu::op::Mul<DType> Op;                    \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "div") {                                     \
      typedef dgl::aten::cpu::op::Div<DType> Op;                    \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_lhs") {                                \
      typedef dgl::aten::cpu::op::CopyLhs<DType> Op;                \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_rhs") {                                \
      typedef dgl::aten::cpu::op::CopyRhs<DType> Op;                \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;     \
    }                                                               \
  } while (0)

}  // namespace op

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SPMM_H_
