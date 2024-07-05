#pragma once

#include <optix_types.h>

namespace Program
{
    static constexpr unsigned int maxTraceDepth{ 1 };
    static constexpr unsigned int maxDCDepth{ 1 };
    static constexpr unsigned int maxCCDepth{ 0 };

    static constexpr unsigned int payloadValueCount{ 7 };
    static constexpr unsigned int payloadSemantics[payloadValueCount]
    {
        //Position
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
        
        //Normal
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
        
        //Misc
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
        OPTIX_PAYLOAD_SEMANTICS_CH_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_MS_WRITE          |
        OPTIX_PAYLOAD_SEMANTICS_AH_NONE           |
        OPTIX_PAYLOAD_SEMANTICS_IS_NONE,
    };

    static constexpr char raygenName[]{ "__raygen__main" };
    static constexpr char missName[]{ "__miss__miss" };
    static constexpr char closehitTriangleName[]{ "__closesthit__triangle" };
    static constexpr char intersectionDiskName[]{ "__intersection__disk" };
    static constexpr char closehitDiskName[]{ "__closesthit__disk" };
    static constexpr char callableName[]{ "__direct_callable__ConductorBxDF" };
}
