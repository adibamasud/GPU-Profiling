# GAPP
GNN-Assisted Pipeline Partitioning using Cross-Device Performance Prediction

   Our work introduces a novel approach using Graph Neural Networks (GNNs) to predict the execution time of neural network computational blocks on resource-constrained target devices (like Raspberry Pi) by leveraging features extracted in a different, more powerful environment (e.g., a server). These performance predictions are then utilized to guide effective pipeline partitioning for distributed DNN inference.

## Overview of the Approach

Our system consists of three main stages:

1.  **Hardware-Independent Feature Extraction:** For a given neural network block (e.g., a specific layer, a sequence of layers, or a standard architectural unit like an Inception module or ResNet block), we use `torch.fx` to obtain its computational graph. We then extract hardware-independent features such as:
    *   Graph structure (nodes, edges, op types)
    *   Module/operation parameters (channels, kernel sizes, etc.)
    *   Tensor shapes at each node (derived via a symbolic forward pass)
    *   Theoretical FLOPs and memory access patterns
    This stage is performed on a **server environment** (see `feature_extractor.py`).

2.  **Target Device Time Measurement:** The same set of defined neural network blocks are executed on the **target resource-constrained device** (e.g., Raspberry Pi CPU) to collect actual wall-clock execution times. This provides the ground truth for our GNN model (see `time_measurer.py`).

3.  **GNN-based Performance Prediction & Pipeline Partitioning:**
    *   A Graph Neural Network is trained using the data collected: (hardware-independent features from Server, actual execution time from Target Device).
    *   The trained GNN can then predict the execution time of *new, unseen* neural network blocks on the target device, using only features extracted on the server.
    *   These predictions, along with a model for inter-device communication costs, are used by a partitioning algorithm to determine an efficient way to split a larger neural network model across the server and the target device for pipelined execution.

## How to Run

**To profile block-level execution time for a particular model on CPU:**

python3 time_measurer.py --output_dir ./results --model_family inception_v3 --batch_size 1 --num_classes 10 --compare_sum_vs_full

Note: 
    * control batch size with --batch_size
    * set --num_classes 10 to make the models compatible with CIFAR10 dataset
    * --compare_sum_vs_full compares the sum of average execution time of each block with the execution time of the full model. 

There will be discrepancy between sum of blocks vs. full model execution time due to CPU caching and memory pressure. 
Sum of blocks: Each block and its I/O tensors have a relatively small memory footprint. They can easily fit into the CPU's fast L1/L2/L3 caches. The memory controller is not heavily stressed.
Full model: The entire model's weights, plus all the intermediate activation tensors, create a much larger total memory "working set." If this set is larger than the CPU's L3 cache, the CPU has to constantly go to the much slower main system RAM. This can create a bottleneck at the memory controller, slowing down the entire process.

**To profile block-level execution time for a particular model on GPU:**

python3 time_measurer.py --output_dir ./results --model_family inception_v3 --device cuda --batch_size 1 --num_classes 10 --compare_sum_vs_full --adjust_for_overhead

Note:
    * "--device cuda" ensures execution on GPU if available
    * control batch size with --batch_size
    * set --num_classes 10 to make the models compatible with CIFAR10 dataset
    * --compare_sum_vs_full compares the sum of average execution time of each block with the execution time of the full model.
    * --adjust_for_overhead measures the overhead associated with GPU kernel launch and synchronization, and subtracts the overhead from the sum of block execution times. The adjusted sum will be closer to the execution time of the full model. 

Despite the adjustment for overhead, there will still remain some discrepancy between sum of blocks vs. full model execution time due to memory access and caching. Data has to be moved from global GPU memory to the processors for each block. In a full model run, intermediate results (activations) can often stay in faster cache memory between layers, which is much more efficient. Measuring blocks individually breaks this chain. 
