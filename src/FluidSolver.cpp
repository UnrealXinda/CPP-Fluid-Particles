#include <memory>

#include "FluidSolver.h"

#include "SPHParticles.h"
#include "BaseSolver.h"
#include "BasicSPHSolver.h"
#include "DFSPHSolver.h"
#include "PBDSolver.h"

struct FFluidSolver::FImpl
{
	FImpl(ESPHSolver solverType, int fluidParticleSize)
	{
		switch (solverType)
		{
		case ESPHSolver::WCSPH:
			pSolver = std::make_shared<BasicSPHSolver>(fluidParticleSize);
			break;
		case ESPHSolver::DFSPH:
			pSolver = std::make_shared<DFSPHSolver>(fluidParticleSize);
			break;
		case ESPHSolver::PBD:
			pSolver = std::make_shared<PBDSolver>(fluidParticleSize);
			break;
		}
	}

	std::shared_ptr<BaseSolver> pSolver;
};

FFluidSolver::FFluidSolver(ESPHSolver solverType, int fluidParticleSize)
{
	Impl = new FFluidSolver::FImpl(solverType, fluidParticleSize);
}

FFluidSolver::~FFluidSolver()
{
	delete Impl;
}

std::shared_ptr<class BaseSolver> FFluidSolver::GetSolverPtr() const
{
	return Impl->pSolver;
}

