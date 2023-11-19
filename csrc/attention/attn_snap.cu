__global__ void single_query_cached_kv_attention_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]，最后一个x是vectorize，一个thread fetch一个vector
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size], num_blocks * block_size=seqlen
  const int* __restrict__ head_mapping,   // [num_heads]，q与kv的head map
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq],2d数组，每个子数组是每个seq的存储kv的physical block nums
  const int* __restrict__ context_lens,   // [num_seqs]，每个句子的长度
  const int max_num_blocks_per_seq, //(max(context_lens) + block_size - 1) / block_size 
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride,
  const int kv_block_stride,//类似于pytorch的stride，每个physical block的stride
  const int kv_head_stride) //类似于pytorch的stride，每个head的stride
  // $ head_size 就是qkv的embedding_size
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);// 每个thread_group 处理blocksize中的1个token，每个token又有numheads * headsize个element，每个block有block size个token，
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  //每组thread处理的token数量，最小为1
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;   //一个block负责一个head，headsize*blocksize的数据
  const int num_heads = gridDim.x;//
  const int kv_head_idx = head_mapping[head_idx]; // q head id->kv head id
  const int seq_idx = blockIdx.y ; // y维度的一个block负责一个seq

  // 每个thread group 向量化load&store，这里其实我有点疑问，为什么是以thread group为单位load 16*8=128bit数据，而不是以thread，因为CUDA每个thread一次性最大可以访问128bit数据
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1); 
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  // 1个thread group处理一个head里面的head size
  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;
  // 当前thread所在的thread group
  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  // 当前thread在thread_group内的offset
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;


    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  //每个block x负责一个head，那么这里申请一块shared mem来存每个thread x读到的head size维度数据
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    // 每个thread读取的q vector都放在q_vecs, 求出当前thread处理的q的最后一维的offset=q_ptr + vec_idx * VEC_SIZE
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }