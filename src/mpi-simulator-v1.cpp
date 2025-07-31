#include "common.h"
#include "mpi.h"
#include "quad-tree.h"
#include "timing.h"
#include <algorithm>

void simulateStep(const QuadTree &quadTree,
                  const std::vector<Particle> &particles,
                  std::vector<Particle> &newParticles, StepParameters params,
                  int offset, int subParticles) {
  for (int i=0; i<subParticles; i++) {
    int source = i + offset;
    auto p = particles[source];
    std::vector<Particle> nearby;
    Vec2 force = Vec2(0.0f, 0.0f);

    quadTree.getParticles(nearby, p.position, params.cullRadius);

    for (Particle p1: nearby) {
      force += computeForce(p, p1, params.cullRadius);
    }
    // update particle state using the computed force
    newParticles[i] = updateParticle(p, force, params.deltaTime);
  }
}

void createDatatypeVec2(MPI_Datatype & mpi_vec2) {
  /* Create datatype for Vec2 */
  int mpi_vec2_lengths[2] = {1, 1};
  MPI_Aint displacement[2];
  MPI_Aint base_address;
  struct Vec2 dummy_vec2;
  MPI_Get_address(&dummy_vec2, &base_address);
  MPI_Get_address(&dummy_vec2.x, &displacement[0]);
  MPI_Get_address(&dummy_vec2.y, &displacement[1]);
  displacement[0] = MPI_Aint_diff(displacement[0], base_address);
  displacement[1] = MPI_Aint_diff(displacement[1], base_address);

  MPI_Datatype types[2] = { MPI_FLOAT, MPI_FLOAT };
  MPI_Type_create_struct(2, mpi_vec2_lengths, displacement, types, &mpi_vec2);
  MPI_Type_commit(&mpi_vec2);
}

void createDatatypeParticle(MPI_Datatype & mpi_particle, MPI_Datatype & mpi_vec2) {
  /* Create datatype for particle */
  int mpi_particle_lengths[4] = {1, 1, 1, 1};
  MPI_Aint displacement[4];
  MPI_Aint base_address;
  struct Particle dummy_particle;
  MPI_Get_address(&dummy_particle, &base_address);
  MPI_Get_address(&dummy_particle.id, &displacement[0]);
  MPI_Get_address(&dummy_particle.mass, &displacement[1]);
  MPI_Get_address(&dummy_particle.position, &displacement[2]);
  MPI_Get_address(&dummy_particle.velocity, &displacement[3]);
  displacement[0] = MPI_Aint_diff(displacement[0], base_address);
  displacement[1] = MPI_Aint_diff(displacement[1], base_address);
  displacement[2] = MPI_Aint_diff(displacement[2], base_address);
  displacement[3] = MPI_Aint_diff(displacement[3], base_address);

  MPI_Datatype types[4] = { MPI_INT, MPI_FLOAT, mpi_vec2, mpi_vec2};
  MPI_Type_create_struct(4, mpi_particle_lengths, displacement, types, &mpi_particle);
  MPI_Type_commit(&mpi_particle);
}

int main(int argc, char *argv[]) {
  int pid;
  int nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  // Get total number of processes specificed at start of run
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  StartupOptions options = parseOptions(argc, argv);

  std::vector<Particle> particles, newParticles;
  if (pid == MASTER) {
    loadFromFile(options.inputFile, particles);
  }

  StepParameters stepParams = getBenchmarkStepParams(options.spaceSize);

  // Create datatype for Vec2
  MPI_Datatype mpi_vec2;
  createDatatypeVec2(mpi_vec2);
  // Create datatype for Particle
  MPI_Datatype mpi_particle;
  createDatatypeParticle(mpi_particle, mpi_vec2);

  // initialize parameters
  int aveParticles;
  int extra;
  int offset;
  int mtype;
  int subParticles;
  int numParticles;
  double buildTreeTotalTime = 0.0;
  double simulationStepTotalTime = 0.0;
  MPI_Status status;
  QuadTree tree;

  // Don't change the timeing for totalSimulationTime.
  MPI_Barrier(MPI_COMM_WORLD);
  Timer totalSimulationTimer;
  Timer buildTreeTimer;
  Timer simulationStepTimer;

  if (pid == MASTER) {
    mtype = FROM_MASTER;
    aveParticles = particles.size() / nproc;
    extra = particles.size() % nproc;
    numParticles = particles.size();

    for (int dest = 1; dest < nproc; dest++) {
      MPI_Send(&numParticles, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
    }

    newParticles.resize(numParticles);
  } else if (pid > MASTER) {
    /* Receive signal from master on offset and number of particles to process */
    mtype = FROM_MASTER;

    MPI_Recv(&numParticles, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

    extra = numParticles % nproc;
    aveParticles = numParticles / nproc;
    offset = (pid < extra) ? ((aveParticles + 1) * (pid - 1)) : ((aveParticles + 1) * extra + aveParticles * (pid - extra));
    subParticles = (pid < extra) ? (aveParticles + 1) : aveParticles;

    particles.resize(numParticles);
    newParticles.resize(subParticles);
  }

  for (int i = 0; i < options.numIterations; i++) {
    if (pid == MASTER) {
      /* allocate particles to workers */
      mtype = FROM_MASTER;

      for (int dest = 1; dest < nproc; dest++) {
        MPI_Send(&particles[0], numParticles, mpi_particle, dest, mtype, MPI_COMM_WORLD);
      }

      double startBuild = buildTreeTimer.elapsed();
      QuadTree::buildQuadTree(particles, tree);
      double finishBuild = buildTreeTimer.elapsed();
      buildTreeTotalTime += finishBuild - startBuild;

      double startSimulationStep = simulationStepTimer.elapsed();
      if (extra) {
        simulateStep(tree, particles, newParticles, stepParams, 0, aveParticles + 1);
      } else {
        simulateStep(tree, particles, newParticles, stepParams, 0, aveParticles);
      }
      double finishSimulationStep = simulationStepTimer.elapsed();
      simulationStepTotalTime += finishSimulationStep - startSimulationStep;

      mtype = FROM_WORKER;
      MPI_Status status;

      for (int i=1; i<nproc; i++) {
        int source = i;

        offset = (source < extra) ? ((aveParticles + 1) * (source - 1)) : ((aveParticles + 1) * extra + aveParticles * (source - extra));
        subParticles = (source < extra) ? (aveParticles + 1) : aveParticles;
        // printf("Received results from worker %d for offset=%d and subParticles=%d: status = %d\n", source, offset, subParticles, status.MPI_TAG);
        MPI_Recv(&newParticles[offset], subParticles, mpi_particle, source, mtype, MPI_COMM_WORLD, &status);
        // printf("Received new particles: %ld\n", newParticles.size());
      }

      particles.swap(newParticles);
      // printf("master finish iteration: %d\n", i);
    } else if (pid > MASTER) {
      // printf("mpi started worker %d\n", pid);

      /* Receive signal from master on particles to process */
      mtype = FROM_MASTER;

      MPI_Recv(&particles[0], numParticles, mpi_particle, MASTER, mtype, MPI_COMM_WORLD, &status);

      // printf("Received order from master for offset=%d and subParticles=%d and particles=%ld: status = %d\n", offset, subParticles, particles.size(), status.MPI_TAG);
      double startBuild = buildTreeTimer.elapsed();
      QuadTree::buildQuadTree(particles, tree);
      double finishBuild = buildTreeTimer.elapsed();
      buildTreeTotalTime += finishBuild - startBuild;

      double startSimulationStep = simulationStepTimer.elapsed();
      simulateStep(tree, particles, newParticles, stepParams, offset, subParticles);
      double finishSimulationStep = simulationStepTimer.elapsed();
      simulationStepTotalTime += finishSimulationStep - startSimulationStep;

      // printf("Sending results to master for offset=%d and subParticles=%d and particles=%ld\n", offset, subParticles, newParticles.size());
      mtype = FROM_WORKER;
      MPI_Send(&newParticles[0], subParticles, mpi_particle, MASTER, mtype, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double totalSimulationTime = totalSimulationTimer.elapsed();

  printf("total build tree time in pid %d: %.6fs\n", pid, buildTreeTotalTime);
  printf("total simulation step time in pid %d: %.6fs\n", pid, simulationStepTotalTime);

  if (pid == MASTER) {
    printf("total simulation time: %.6fs\n", totalSimulationTime);
    saveToFile(options.outputFile, particles);
  }

  MPI_Finalize();
}
