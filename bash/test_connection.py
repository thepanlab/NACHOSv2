"""test_connection.py
Run it like this
mpirun --mca btl ^usnic,self,tcp --hostfile hosts.txt python /home/pcallec/NACHOS/bash/mpi_connection_test.py
"""


from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get hostname
import socket
hostname = socket.gethostname()

if rank == 0:
    print(f"Master node running on {hostname} (rank {rank})")
    # Send message to all other processes
    for i in range(1, size):
        comm.send(f"Hello from Master (rank 0) to rank {i}", dest=i, tag=11)
else:
    # Receive message from rank 0
    message = comm.recv(source=0, tag=11)
    print(f"Rank {rank} running on {hostname} received message: '{message}'")

# Finalize MPI
comm.Barrier()
if rank == 0:
    print("MPI connection test completed successfully.")
