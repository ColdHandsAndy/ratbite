#pragma once

#include <cstdint>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "../core/util_macros.h"

// Cycles QRNG approach https://github.com/blender/cycles/tree/main
namespace QRNG
{
	enum class DimensionOffset : uint32_t
	{
		FILTER = 0,
		LENS = 1,

		WAVELENGTH = 0,
		LIGHT = 1,
		SURFACE_BXDF = 2,
		
		DESC = 16,
	};
	CU_DEVICE CU_INLINE uint32_t getPixelHash(uint32_t x, uint32_t y)
	{
		const uint32_t qx{ 1103515245U * ((x >> 1U) ^ (y)) };
		const uint32_t qy{ 1103515245U * ((y >> 1U) ^ (x)) };
		const uint32_t n{ 1103515245U * ((qx) ^ (qy >> 3U)) };
		constexpr uint32_t seed{ 0 };
		return n^seed;
	}
	class State
	{
	private:
		uint32_t m_sample{};
		uint32_t m_offset{};
		uint32_t m_hash{};
	public:
		CU_DEVICE CU_INLINE State(uint32_t sample, uint32_t hash)
			: m_sample{sample}, m_offset{static_cast<uint32_t>(DimensionOffset::DESC)}, m_hash{hash}
		{

		}

		CU_DEVICE CU_INLINE uint32_t getSample() const { return m_sample; }
		CU_DEVICE CU_INLINE uint32_t getOffset() const { return m_offset; }
		CU_DEVICE CU_INLINE uint32_t getHash() const { return m_hash; }

		CU_DEVICE CU_INLINE void advanceSample()
		{
			++m_sample;
			m_offset = static_cast<uint32_t>(DimensionOffset::DESC);
		}
		CU_DEVICE CU_INLINE void advanceBounce()
		{
			m_offset += static_cast<uint32_t>(DimensionOffset::DESC);
		}
	};

	extern __device__ inline uint32_t sobolBurleyTable[4][32];
	class Sobol
	{
	public:
		Sobol() = delete;
		Sobol(Sobol &&) = delete;
		Sobol(const Sobol &) = delete;
		Sobol &operator=(Sobol &&) = delete;
		Sobol &operator=(const Sobol &) = delete;
		~Sobol() = delete;

		CU_DEVICE CU_INLINE static float sample1D(const State& qrngState, DimensionOffset dOffset)
		{
			uint32_t offset{ qrngState.getOffset() + static_cast<uint32_t>(dOffset) };
			uint32_t seed{ qrngState.getHash() ^ getDimHash(offset) };
			uint32_t index{ reversedBitOwen(reverseIntegerBits(qrngState.getSample()), seed ^ 0xbff95bfe) };

			return sobolBurley(index, 0, seed ^ 0x635c77bd);
		}
		CU_DEVICE CU_INLINE static glm::vec2 sample2D(const State& qrngState, DimensionOffset dOffset)
		{
			uint32_t offset{ qrngState.getOffset() + static_cast<uint32_t>(dOffset) };
			uint32_t seed{ qrngState.getHash() ^ getDimHash(offset) };
			uint32_t index{ reversedBitOwen(reverseIntegerBits(qrngState.getSample()), seed ^ 0xf8ade99a) };

			return glm::vec2{ sobolBurley(index, 0, seed ^ 0xe0aaaf76), sobolBurley(index, 1, seed ^ 0x94964d4e) };
		}
		CU_DEVICE CU_INLINE static glm::vec3 sample3D(const State& qrngState, DimensionOffset dOffset)
		{
			uint32_t offset{ qrngState.getOffset() + static_cast<uint32_t>(dOffset) };
			uint32_t seed{ qrngState.getHash() ^ getDimHash(offset) };
			uint32_t index{ reversedBitOwen(reverseIntegerBits(qrngState.getSample()), seed ^ 0xcaa726ac) };

			return glm::vec3{ sobolBurley(index, 0, seed ^ 0x9e78e391), sobolBurley(index, 1, seed ^ 0x67c33241), sobolBurley(index, 2, seed ^ 0x78c395c5) };
		}
	private:
		CU_DEVICE CU_INLINE static uint32_t getDimHash(uint32_t offset)
		{
			uint32_t i{ offset };
			i ^= i >> 16;
			i *= 0x21f0aaad;
			i ^= i >> 15;
			i *= 0xd35a2d97;
			i ^= i >> 15;
			return i ^ 0xe6fe3beb;
		}
		CU_DEVICE CU_INLINE static uint32_t reverseIntegerBits(uint32_t x)
		{
			return __brev(x);
		}
		CU_DEVICE CU_INLINE static uint32_t countLeadingZeros(uint32_t x)
		{
			return __clz(x);
		}
		CU_DEVICE CU_INLINE static uint32_t reversedBitOwen(uint32_t n, uint32_t seed)
		{
			n ^= n * 0x3d20adea;
			n += seed;
			n *= (seed >> 16) | 1;
			n ^= n * 0x05526c56;
			n ^= n * 0x53a22864;
			return n;
		}
		CU_DEVICE CU_INLINE static float uintToFloatExclusive(uint32_t x)
		{
			return static_cast<float>(x) * (1.0f / 4294967808.0f);
		}
		CU_DEVICE CU_INLINE static float sobolBurley(
				uint32_t revBitIndex,
				const uint32_t dimension,
				const uint32_t scrambleSeed)
		{
			uint32_t result{ 0 };

			if (dimension == 0)
			{
				result = reverseIntegerBits(revBitIndex);
			}
			else
			{
				uint32_t i{ 0 };
				while (revBitIndex != 0)
				{
					uint32_t j{ countLeadingZeros(revBitIndex) };
					result ^= sobolBurleyTable[dimension][i + j];
					i += j + 1;

					revBitIndex <<= j;
					revBitIndex <<= 1;
				}
			}

			result = reverseIntegerBits(reversedBitOwen(result, scrambleSeed));

			return uintToFloatExclusive(result);
		}
	};

	__device__ inline uint32_t sobolBurleyTable[4][32] = 
	{
		{
			0x00000001, 0x00000002, 0x00000004, 0x00000008,
			0x00000010, 0x00000020, 0x00000040, 0x00000080,
			0x00000100, 0x00000200, 0x00000400, 0x00000800,
			0x00001000, 0x00002000, 0x00004000, 0x00008000,
			0x00010000, 0x00020000, 0x00040000, 0x00080000,
			0x00100000, 0x00200000, 0x00400000, 0x00800000,
			0x01000000, 0x02000000, 0x04000000, 0x08000000,
			0x10000000, 0x20000000, 0x40000000, 0x80000000,
		},
		{
			0x00000001, 0x00000003, 0x00000005, 0x0000000f,
			0x00000011, 0x00000033, 0x00000055, 0x000000ff,
			0x00000101, 0x00000303, 0x00000505, 0x00000f0f,
			0x00001111, 0x00003333, 0x00005555, 0x0000ffff,
			0x00010001, 0x00030003, 0x00050005, 0x000f000f,
			0x00110011, 0x00330033, 0x00550055, 0x00ff00ff,
			0x01010101, 0x03030303, 0x05050505, 0x0f0f0f0f,
			0x11111111, 0x33333333, 0x55555555, 0xffffffff,
		},
		{
			0x00000001, 0x00000003, 0x00000006, 0x00000009,
			0x00000017, 0x0000003a, 0x00000071, 0x000000a3,
			0x00000116, 0x00000339, 0x00000677, 0x000009aa,
			0x00001601, 0x00003903, 0x00007706, 0x0000aa09,
			0x00010117, 0x0003033a, 0x00060671, 0x000909a3,
			0x00171616, 0x003a3939, 0x00717777, 0x00a3aaaa,
			0x01170001, 0x033a0003, 0x06710006, 0x09a30009,
			0x16160017, 0x3939003a, 0x77770071, 0xaaaa00a3,
		},
		{
			0x00000001, 0x00000003, 0x00000004, 0x0000000a,
			0x0000001f, 0x0000002e, 0x00000045, 0x000000c9,
			0x0000011b, 0x000002a4, 0x0000079a, 0x00000b67,
			0x0000101e, 0x0000302d, 0x00004041, 0x0000a0c3,
			0x0001f104, 0x0002e28a, 0x000457df, 0x000c9bae,
			0x0011a105, 0x002a7289, 0x0079e7db, 0x00b6dba4,
			0x0100011a, 0x030002a7, 0x0400079e, 0x0a000b6d,
			0x1f001001, 0x2e003003, 0x45004004, 0xc900a00a,
		},
	};
}
