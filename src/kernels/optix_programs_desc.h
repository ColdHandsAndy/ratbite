#pragma once

#include <optix_types.h>

namespace Program
{
    static constexpr unsigned int maxTraceDepth{ 1 };
    static constexpr unsigned int maxDCDepth{ 1 };
    static constexpr unsigned int maxCCDepth{ 0 };

    static constexpr unsigned int payloadValueCount{ 12 };
    static constexpr unsigned int payloadSemantics[payloadValueCount]
    {
        // Position
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,

        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,

        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,

        // Encoded geometry normal
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,

		// Primitive (Light) index
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,

		// Barycentrics
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_NONE,

		OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
		OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
		OPTIX_PAYLOAD_SEMANTICS_MS_WRITE          |
		OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
		OPTIX_PAYLOAD_SEMANTICS_IS_NONE,

		// Light Type + Material Index
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_WRITE,

		// Shading frame
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_NONE,

        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_NONE,

        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_NONE,

        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_NONE,
    };

    static constexpr char raygenName[]{ "__raygen__main" };
    static constexpr char missName[]{ "__miss__miss" };
    static constexpr char closehitTriangleName[]{ "__closesthit__triangle" };
    static constexpr char intersectionDiskName[]{ "__intersection__disk" };
	static constexpr char closehitDiskName[]{ "__closesthit__disk" };
	static constexpr char closehitSphereName[]{ "__closesthit__sphere" };
    static constexpr char pureConductorBxDFName[]{ "__direct_callable__PureConductorBxDF" };
    static constexpr char pureDielectricBxDFName[]{  "__direct_callable__PureDielectricBxDF" };
    static constexpr char complexSurfaceBxDFName[]{  "__direct_callable__ComplexSurface_BxDF" };
}
