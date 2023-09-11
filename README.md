# CDAC_SpMV_Project
This reposirtory contains programs and documentation of comparison study of SpMV multiplication.

Sparse matrix-vector multiplication (SpMV) is one of the most important compu-
tational kernels in various computational intensive areas. SpMV is the multiplication of
sparse matrix A with vector x given by Ax = b. Here we have performed a compari-
son study between (I) naive approach, (II) Intel oneAPI library calls and (III) CUDA
enabled NVIDIA libraries. The study was carried out on matrices with single and dou-
ble prescision random numbers with increasing matrix sizes. Compressed Sparse Row
(CSR) format was used for sparse matrix storage. We found that on CPU side Intel
MKL’s SpMV library calls gives best performance. On GPU side slight difference is ob-
served. CSR kernel and cuSPARSE shows similar performance on both single and double
precision.
