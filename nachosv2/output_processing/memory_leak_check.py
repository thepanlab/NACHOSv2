import gc
import tracemalloc


def initiate_memory_leak_check():
    """
    Initializes memory leak checking by starting garbage collection and memory tracking.

    Returns:
        snapshot1 (tracemalloc.Snapshot): A snapshot of the current memory allocation for comparison.
    """
    
    # Forces garbage collection to free unused memory
    gc.collect()
    
    # Starts memory tracking
    tracemalloc.start()
    
    # Takes an initial snapshot of memory allocation
    snapshot1 = tracemalloc.take_snapshot()
    
    return snapshot1
    
    
def end_memory_leak_check(snapshot1):
    """
    Ends the memory leak check by taking a final snapshot and comparing it with the initial snapshot.

    Args:
        snapshot1 (tracemalloc.Snapshot): The initial snapshot taken before the memory-intensive operation.
    """
    
    # Forces garbage collection again to free unused memory
    gc.collect()
    
    
    # Takes a final snapshot of memory allocation
    snapshot2 = tracemalloc.take_snapshot()
    
    
    # Compares the final snapshot with the initial one
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    
    # Calculates and prints the total memory not freed
    total = sum(stat.size_diff for stat in top_stats)
    
    print(f"Total memory not freed: {total / 1024 / 1024:.2f} MiB")

    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)
        