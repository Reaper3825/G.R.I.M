//======================================================//
//  Diagnostics.hpp
//  Umbrella include for training diagnostic helpers.
//======================================================//

#pragma once

#include "DiagnosticInference.hpp"
#include "RhoDiagnostic.hpp"
#include "LMHeadWeightStats.hpp"
#include "LogitScaleDiagnostic.hpp"
#include "BoundaryDiagnostic.hpp"
#include "SpecialTokenDiagnostic.hpp"
#include "AtomStatsDiagnostic.hpp"
#include "LossBaselineDiagnostic.hpp"
#include "LossStatsDiagnostic.hpp"
#include "GradientNormDiagnostic.hpp"
#include "PostClipParamGradEmbLmEquation.hpp"
#include "OptimizerStepGuards.hpp"
#include "TieVerifyDiagnostic.hpp"
#include "OptimizerMomentDiagnostic.hpp"
#include "PostOptimizerWeightTrace.hpp"
#include "PredictionDistributionDiagnostic.hpp"
#include "DiagnosticGates.hpp"
#include "HistogramDiagnostic.hpp"
