## Calibration
### Goal 1
Write a small callibration program that does the following
- Identify all backends 
- For each backend
  - Identify its capabilities
  - GPU Type
  	- CUDA
		- CUDA ARCH (Supported or not within our system
		- MEMORY
		- CUDA CORES
		- TENSOR CORES
		- MEMORY BANDWIDTH
		- ...
	- VULKAN
		- similar
  - CPU
	- MEMORY (DD4/DDR5/HBM2..)
	- CPU (...)
	- CPU FLAGS
	- ...
  - Storage
	-NVME...
	
  - PCIE
	- PCIE-lane width between each backend.
	
 - For every backend, create a score/kpi (Figure out a method) - and we will sort it. This will include the PCIE-lane bandwidth capacity.
 
### Goal 2
For a given model that will be taken as -m <model name parameter>, identify the numbe of experts/layer, tensors, quantization etc... and we will fit the number of experts that will reside on each backend. 80% must be cached. 10% working mem, and 10% for OS/Kernel related stuff.

### Goal 3
Create a hpcache-callibration.json file, that will be passed to the llama-cpp engine.

## llama.cpp HPCache Proof of Concept.
- Introduce a --hpcache option, and use the hpcache-callibration.json file, and fit the parameters.
- Ensure that each backend is loaded and can execute. Since  llama.cpp already provides backend for all, we only schedule.
- Note we are not doing any Page cache here. (that would be susequent).
- Every token through the decoding phase and results is merge-sorted
- Result needs to be the same as it would have been on GPU only.
- Current GPU only runs at 2/sec token generation, Strata runs at 30 t/s. 
- You can see what Strata does - it is in this directory : C:\Users\suresh\devspace\github\Strata\. It uses llama.cpp, but does something more to it.






