#pragma once



#define ARRAYSIZE(x) ((sizeof(x) / sizeof(*(x))) / static_cast<size_t>(!(sizeof(x) % sizeof(*(x)))))
#define ALIGNED_SIZE(size, alignment_requirement) (size + ((alignment_requirement - (size % alignment_requirement)) % alignment_requirement))
#define DISPATCH_SIZE(dispatchElementCount, groupElementCount)	\
	(dispatchElementCount / groupElementCount + 				\
	((dispatchElementCount % groupElementCount) != 0 ? 1 : 0))


#if defined(__CUDACC__)
#    define CU_HOSTDEVICE __host__ __device__
#    define CU_HOST __host__
#    define CU_DEVICE __device__
#    define CU_INLINE __forceinline__
#else
#    define CU_HOSTDEVICE
#    define CU_HOST
#    define CU_DEVICE
#    define CU_INLINE inline
#endif

#if defined(__CUDACC__)
#    define CUPTR(type) type ## *
#else
#    define CUPTR(type) CUdeviceptr
#endif


#define STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_ONE_OP(flagsType, enumClass, operation)		\
	flagsType operator ## operation ## (enumClass b0, enumClass b1)						\
	{ return static_cast<flagsType>(b0) operation static_cast<flagsType>(b1); }			\
	flagsType operator ## operation ## (flagsType f, enumClass b)						\
	{ return static_cast<flagsType>(f) operation static_cast<flagsType>(b);	  }			\
	flagsType operator ## operation ## (enumClass b, flagsType f)						\
	{ return static_cast<flagsType>(f) operation static_cast<flagsType>(b);	  }
#define STRONGLY_TYPED_ENUM_OPERATOR_EXPAND(flagsType, enumClass) \
	STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_ONE_OP(flagsType, enumClass, |) \
	STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_ONE_OP(flagsType, enumClass, &) \
	STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_ONE_OP(flagsType, enumClass, ^)
#define STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_ONE_OP_WITH_PREFIX(flagsType, enumClass, operation, prefix)		\
	prefix flagsType operator ## operation ## (enumClass b0, enumClass b1)									\
	{ return static_cast<flagsType>(b0) operation static_cast<flagsType>(b1); }								\
	prefix flagsType operator ## operation ## (flagsType f, enumClass b)									\
	{ return static_cast<flagsType>(f) operation static_cast<flagsType>(b);	  }								\
	prefix flagsType operator ## operation ## (enumClass b, flagsType f)									\
	{ return static_cast<flagsType>(f) operation static_cast<flagsType>(b);	  }
#define STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_WITH_PREFIX(flagsType, enumClass, prefix) \
	STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_ONE_OP_WITH_PREFIX(flagsType, enumClass, |, prefix) \
	STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_ONE_OP_WITH_PREFIX(flagsType, enumClass, &, prefix) \
	STRONGLY_TYPED_ENUM_OPERATOR_EXPAND_ONE_OP_WITH_PREFIX(flagsType, enumClass, ^, prefix)
