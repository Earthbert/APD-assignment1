
# Homework 1 - APD

## Daraban Albert-Timotei

The first thing I have done is to measure the time each of the 5 steps in the sequential solution take on the biggest test available.

The results in reverse order were:

- Rescale: 1s-2s
- March the squares: ~0.01s
- Write output: ~0.005s
- Sample the grid: ~0.001s
- Initialize contour map: ~0.0001s

After looking at the code, I have discovered that "rescale," "march," and "sample" could be easily parallelized using the "start-end" technique learned at the lab.

Writing the output would be too hard (maybe impossible; I am not sure) to share between threads, and even if I did that, it wouldn't help because the bottleneck is in the operating system, which performs the actual writing to the disk (needs citation).

Initializing the contour map is a constant operation that takes far too little time to even consider sharing the load between threads.
