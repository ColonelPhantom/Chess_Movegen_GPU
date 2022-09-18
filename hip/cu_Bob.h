#pragma once
#include <hip/hip_runtime.h>

#pragma once

// Robert Hyatt's and Michael Sherwin's classical bitboard approach
// generate moves for the sliding pieces.

//Cuda Translation by Daniel Inf�hr - Jan. 2022
//Contact: daniel.infuehr@live.de

#include <cstdint>
#include <array>
#include <immintrin.h>
#include "cu_Common.h"

namespace BobLU {

	struct Rays {
		uint64_t rayNW;
		uint64_t rayNN;
		uint64_t rayNE;
		uint64_t rayEE;
		uint64_t raySE;
		uint64_t raySS;
		uint64_t raySW;
		uint64_t rayWW;
		uint64_t rwsNW;
		uint64_t rwsNN;
		uint64_t rwsNE;
		uint64_t rwsEE;
		uint64_t rwsSE;
		uint64_t rwsSS;
		uint64_t rwsSW;
		uint64_t rwsWW;

		uint64_t queen;
	};

	constexpr Rays Initialize(int sq) {
		enum { FILE1, FILE2, FILE3, FILE4, FILE5, FILE6, FILE7, FILE8 };
		enum { RANK1, RANK2, RANK3, RANK4, RANK5, RANK6, RANK7, RANK8 };

		int ts{}, file{}, rank{}, c{};
		Rays ray = {};
		file = sq & 7;
		rank = sq >> 3;

		// Northwest
		ray.rayNW = 0;
		for (c = 1, ts = sq + 7; file - c >= FILE1 && rank + c <= RANK8; c++, ts += 7) ray.rayNW |= 1ull << ts;
		ray.rwsNW = ray.rayNW | 0x8000000000000000;

		// Northeast
		ray.rayNE = 0;
		for (c = 1, ts = sq + 9; file + c <= FILE8 && rank + c <= RANK8; c++, ts += 9) ray.rayNE |= 1ull << ts;
		ray.rwsNE = ray.rayNE | 0x8000000000000000;

		// Southeast
		ray.raySE = 0;
		for (c = 1, ts = sq - 7; file + c <= FILE8 && rank - c >= RANK1; c++, ts -= 7) ray.raySE |= 1ull << ts;
		ray.rwsSE = ray.raySE | 0x0000000000000001;

		// Southwest
		ray.raySW = 0;
		for (c = 1, ts = sq - 9; file - c >= FILE1 && rank - c >= RANK1; c++, ts -= 9) ray.raySW |= 1ull << ts;
		ray.rwsSW = ray.raySW | 0x0000000000000001;

		// North
		ray.rayNN = 0;
		for (c = 1, ts = sq + 8; rank + c <= RANK8; c++, ts += 8) ray.rayNN |= 1ull << ts;
		ray.rwsNN = ray.rayNN | 0x8000000000000000;

		// East
		ray.rayEE = 0;
		for (c = 1, ts = sq + 1; file + c <= FILE8; c++, ts += 1) ray.rayEE |= 1ull << ts;
		ray.rwsEE = ray.rayEE | 0x8000000000000000;

		// South
		ray.raySS = 0;
		for (c = 1, ts = sq - 8; rank - c >= RANK1; c++, ts -= 8) ray.raySS |= 1ull << ts;
		ray.rwsSS = ray.raySS | 0x0000000000000001;

		// West
		ray.rayWW = 0;
		for (c = 1, ts = sq - 1; file - c >= FILE1; c++, ts -= 1) ray.rayWW |= 1ull << ts;
		ray.rwsWW = ray.rayWW | 0x0000000000000001;


		ray.queen = ray.rayNN | ray.rayEE | ray.raySS | ray.rayWW |
					ray.rayNW | ray.rayNE | ray.raySE | ray.raySW;
		return ray;
	}
	__shared__ Rays ray_share[64];

	__inline__ __device__ void Prepare(int threadIdx)
	{
		if (threadIdx < 64)
		{
			ray_share[threadIdx] = Initialize(threadIdx);
		}
		__syncthreads();
	}

	__device__ uint64_t countr_zero(uint64_t value)
	{
		return __clzll(value);
	}

	__device__ uint64_t countl_zero(uint64_t value)
	{
		return __ffsll((unsigned long long)value);
	}

	__device__ uint64_t Queen(int sq, uint64_t occ) {
		occ |= 0x8000000000000001;
		const Rays& r = ray_share[sq];

		uint64_t bb = (
			  ray_share[countr_zero(r.rwsNW & occ)].rayNW
			| ray_share[countr_zero(r.rwsNN & occ)].rayNN
			| ray_share[countr_zero(r.rwsNE & occ)].rayNE
			| ray_share[countr_zero(r.rwsEE & occ)].rayEE
			| ray_share[countl_zero(r.rwsSE & occ)].raySE
			| ray_share[countl_zero(r.rwsSS & occ)].raySS
			| ray_share[countl_zero(r.rwsSW & occ)].raySW
			| ray_share[countl_zero(r.rwsWW & occ)].rayWW)
			^ r.queen;

		return bb; // Rook(sq, occ) | Bishop(sq, occ);
	}


}