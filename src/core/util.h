#pragma once

#define ARRAYSIZE(x) ((sizeof(x) / sizeof(*(x))) / static_cast<size_t>(!(sizeof(x) % sizeof(*(x)))))
#define ALIGNED_SIZE(size, alignment_requirement) (size + ((alignment_requirement - (size % alignment_requirement)) % alignment_requirement))
#define DISPATCH_SIZE(dispatchElementCount, groupElementCount)	\
	(dispatchElementCount / groupElementCount + 				\
	((dispatchElementCount % groupElementCount) != 0 ? 1 : 0))
#define CONTAINER_ELEMENT_SIZE(x) sizeof(decltype(x)::value_type)


#if defined(__CUDACC__)
#    define CU_HOSTDEVICE __host__ __device__
#    define CU_HOST __host__
#    define CU_DEVICE __device__
#    define CU_INLINE __forceinline__
#	 define CU_CONSTANT __constant__
#else
#    define CU_HOSTDEVICE
#    define CU_HOST
#    define CU_DEVICE
#    define CU_INLINE inline
#	 define CU_CONSTANT
#endif

#if defined(__CUDACC__)
#    define CUPTR(type) type ## *
#else
#    define CUPTR(type) CUdeviceptr
#endif

#if defined(__CUDACC__)
#include <cuda/std/concepts>
template<typename T, typename... U>
	concept AllSameAs = (cuda::std::same_as<T, U> && ...);
#else
#include <concepts>
template<typename T, typename... U>
	concept AllSameAs = (std::same_as<T, U> && ...);
#endif


namespace
{
#if defined(__CUDACC__)
	using namespace cuda::std;
#else
	using namespace std;
#endif
	
	template<typename enumType>
		struct enumBitwiseEnabled
		{
			static constexpr bool value{ false };
		};
	template <typename enumType>
		constexpr bool enumBitwiseEnabledV = enumBitwiseEnabled<enumType>::value;

	template <typename enumType>
		CU_DEVICE CU_INLINE constexpr auto operator|(const enumType& lhs, const enumType& rhs)
		-> typename enable_if_t<enumBitwiseEnabledV<enumType>, enumType>
		{
			return static_cast<enumType>(
					static_cast<underlying_type_t<enumType>>(lhs) |
					static_cast<underlying_type_t<enumType>>(rhs));
		}
	template <typename enumType>
		CU_DEVICE CU_INLINE constexpr auto operator|=(enumType& lhs, const enumType& rhs)
		-> typename enable_if_t<enumBitwiseEnabledV<enumType>, void>
		{
			lhs = lhs | rhs;
		}

	template <typename enumType>
		CU_DEVICE CU_INLINE constexpr auto operator&(const enumType& lhs, const enumType& rhs)
		-> typename enable_if_t<enumBitwiseEnabledV<enumType>, enumType>
		{
			return static_cast<enumType>(
					static_cast<underlying_type_t<enumType>>(lhs) &
					static_cast<underlying_type_t<enumType>>(rhs));
		}
	template <typename enumType>
		CU_DEVICE CU_INLINE constexpr auto operator&=(enumType& lhs, const enumType& rhs)
		-> typename enable_if_t<enumBitwiseEnabledV<enumType>, void>
		{
			lhs = lhs & rhs;
		}

	template <typename enumType>
		CU_DEVICE CU_INLINE constexpr auto operator^(const enumType& lhs, const enumType& rhs)
		-> typename enable_if_t<enumBitwiseEnabledV<enumType>, enumType>
		{
			return static_cast<enumType>(
					static_cast<underlying_type_t<enumType>>(lhs) ^
					static_cast<underlying_type_t<enumType>>(rhs));
		}
	template <typename enumType>
		CU_DEVICE CU_INLINE constexpr auto operator^=(enumType& lhs, const enumType& rhs)
		-> typename enable_if_t<enumBitwiseEnabledV<enumType>, void>
		{
			lhs = lhs ^ rhs;
		}

	template <typename enumType>
		CU_DEVICE CU_INLINE constexpr auto operator~(const enumType& s)
		-> typename enable_if_t<enumBitwiseEnabledV<enumType>, enumType>
		{
			return static_cast<enumType>(~static_cast<underlying_type_t<enumType>>(s));
		}
}

#if defined(__CUDACC__)
#define IS_ENUM(enumType) cuda::std::is_enum<enumType>::value
#else
#define IS_ENUM(enumType) std::is_enum<enumType>::value
#endif

#define ENABLE_ENUM_BITWISE_OPERATORS(enumType) \
		template<> \
		struct enumBitwiseEnabled<enumType> \
		{ \
			static_assert(IS_ENUM(enumType), "Type must be enum"); \
			static constexpr bool value{ true }; \
		};


#include <filesystem>
std::filesystem::path getExePath();
std::filesystem::path getExeDir();
