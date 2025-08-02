#include <iostream>
#include <cstdlib> // For atoi, rand, srand
#include <ctime>   // For time
#include <mpi.h>

void walker_process();
void controller_process();

int domain_size;
int max_steps;
int world_rank;
int world_size;

int main(int argc, char **argv)
{
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes and the rank of this process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3)
    {
        if (world_rank == 0)
        {
            std::cerr << "Usage: mpirun -np <p> " << argv[0] << " <domain_size> <max_steps>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    domain_size = atoi(argv[1]);
    max_steps = atoi(argv[2]);

    if (world_rank == 0)
    {
        // Rank 0 is the controller
        controller_process();
    }
    else
    {
        // All other ranks are walkers
        walker_process();
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

void walker_process()
{
    // Seed the random number generator uniquely per rank
    srand(time(NULL) + world_rank);

    int position = 0;

    for (int step = 1; step <= max_steps; ++step)
    {
        // Randomly move left (-1) or right (+1)
        int move = (rand() % 2 == 0) ? -1 : 1;
        position += move;

        // Check if walker has moved outside the domain
        if (position < -domain_size || position > domain_size)
        {
            std::cout << "Rank " << world_rank << ": Walker finished in " << step << " steps." << std::endl;

            // Send completion message to controller
            MPI_Send(&step, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            break;
        }

        // If max_steps reached and still inside the domain, will handle below
    }

    // If walker has not moved outside domain after max_steps
    if (position >= -domain_size && position <= domain_size)
    {
        std::cout << "Rank " << world_rank << ": Walker finished in " << max_steps << " steps." << std::endl;

        // Send completion message to controller
        MPI_Send(&max_steps, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

void controller_process()
{
    int num_walkers = world_size - 1;
    int finished_count = 0;

    for (int i = 0; i < num_walkers; ++i)
    {
        int steps_finished;
        MPI_Status status;

        // Receive a message from any walker signaling completion
        MPI_Recv(&steps_finished, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

        int sender_rank = status.MPI_SOURCE;
        std::cout << "Controller: Received finished message from Rank " << sender_rank
                  << " (completed in " << steps_finished << " steps)." << std::endl;
        finished_count++;
    }

    std::cout << "Controller: All " << num_walkers << " walkers have finished." << std::endl;
}
