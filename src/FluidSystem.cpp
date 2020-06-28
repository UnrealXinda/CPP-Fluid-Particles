#include <memory>

#include "FluidSystem.h"
#include "FluidSolver.h"
#include "FluidParticles.h"

#include "SPHSystem.h"
#include "SPHParticles.h"
#include "BaseSolver.h"

struct FFluidSystem::FImpl
{
	FImpl(
		std::shared_ptr<SPHParticles>& fluidParticles,
		std::shared_ptr<SPHParticles>& boundaryParticles,
		std::shared_ptr<BaseSolver>&   solver,
		const FFluidSystemConfig&      config) :
		pSystem(
			std::make_shared<SPHSystem>(
				fluidParticles,
				boundaryParticles,
				solver,
				make_float3(config.SpaceSizeY, config.SpaceSizeZ, config.SpaceSizeX), // Convert from Z Up to Y Up
				config.CellLength,
				config.SmoothingRadius,
				config.TimeStep,
				config.M0,
				config.Rho0,
				config.RhoBoundary,
				config.Stiff,
				config.Visc,
				config.SurfaceTensionIntensity,
				config.AirPressure,
				make_float3(config.GY, config.GZ, config.GX), // Convert from Z Up to Y Up
				make_int3(config.CellSizeY, config.CellSizeZ, config.CellSizeX)) // Convert from Z Up to Y Up
		)
	{}

	std::shared_ptr<SPHSystem> pSystem;
};

FFluidSystem::FFluidSystem(
	const FSPHParticles&      FluidParticles,
	const FBoundaryParticles& BoundaryParticles,
	const FFluidSolver&       FluidSolver,
	const FFluidSystemConfig& Config)
{
	Impl = new FFluidSystem::FImpl
	(
		FluidParticles.GetParticlesPtr(),
		BoundaryParticles.GetParticlesPtr(),
		FluidSolver.GetSolverPtr(),
		Config
	);
}

FFluidSystem::~FFluidSystem()
{
	delete Impl;
}

float FFluidSystem::Step()
{
	return Impl->pSystem->step();
}

int FFluidSystem::Size() const
{
	return Impl->pSystem->size();
}

int FFluidSystem::FluidSize() const
{
	return Impl->pSystem->fluidSize();
}

int FFluidSystem::BoundarySize() const
{
	return Impl->pSystem->boundarySize();
}

int FFluidSystem::TotalSize() const
{
	return Impl->pSystem->totalSize();
}
