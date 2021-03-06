#pragma once

class FSPHParticles final
{
public:

	FSPHParticles(float* Pos, int Size);
	~FSPHParticles();

	FSPHParticles(const FSPHParticles&) = delete;
	FSPHParticles& operator=(const FSPHParticles&) = delete;

	friend class FFluidSystem;

	int Size() const;

	void CopyBackParticlePositions(float* Dst);
	void CopyBackParticleTransforms(float* Dst);

private:

	struct FImpl;
	FImpl* Impl;

	std::shared_ptr<class SPHParticles> GetParticlesPtr() const;
};

class FBoundaryParticles final
{
public:

	FBoundaryParticles(float* Pos, int Size);
	~FBoundaryParticles();

	FBoundaryParticles(const FBoundaryParticles&) = delete;
	FBoundaryParticles& operator=(const FBoundaryParticles&) = delete;

	friend class FFluidSystem;

	int Size() const;

private:

	struct FImpl;
	FImpl* Impl;

	std::shared_ptr<class SPHParticles> GetParticlesPtr() const;
};