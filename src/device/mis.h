#pragma once

#include <cuda_runtime.h>

#include "../core/util.h"

namespace MIS
{
    CU_DEVICE CU_INLINE float balanceHeuristic(int nf, float fPDF, int ng, float gPDF)
    {
        return (nf * fPDF) / (nf * fPDF + ng * gPDF);
    }

    CU_DEVICE CU_INLINE float powerHeuristic(int nf, float fPDF, int ng, float gPDF)
    {
        float f{ nf * fPDF };
        float g{ ng * gPDF };
        float sqrF{ f * f };
        float sqrG{ g * g };
        return sqrF / (sqrF + sqrG);
    }
}
