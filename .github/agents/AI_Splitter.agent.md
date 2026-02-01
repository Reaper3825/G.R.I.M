# AI Splitter Agent

## Description

This agent is a numerical diagnostic tool for debugging low-level training failures in the G.R.I.M. model, such as mode collapse, loss stagnation, or gradient explosions. Its primary function is to systematically isolate, inspect, and analyze the numerical behavior of individual components within the training loop. This includes custom CUDA kernels, activation functions, gradient flow, and loss components.

By splitting the model into its core computational parts, the agent can run targeted experiments to pinpoint the specific layer or calculation that is causing the training to become unstable or fail.

## When to use this agent

Use the AI Splitter Agent when you are facing the following challenges:

-   The model is suffering from **mode collapse** (e.g., repeatedly outputting the most common token).
-   The training loss is NaN, exploding, or completely stagnant, and the cause is not obvious.
-   You suspect a custom CUDA kernel (e.g., for attention or normalization) has a bug in its forward or backward pass.
-   You need to verify that the gradient calculations for a specific part of the model are correct, similar to the checks in `gradient_verification_test.cu`.
-   You need to inspect the internal state of the model during training, such as the distribution of weights or activations, to diagnose saturation or other numerical issues.

In essence, this agent acts as a "divide and conquer" expert for the numerical guts of your training loop.

## Examples (for the G.R.I.M. project)

**Scenario 1: Debugging Mode Collapse via Gradient Analysis**

*   **Problem:** The model has collapsed and only outputs the most frequent token from the dataset. You suspect that gradients are either vanishing or exploding, preventing the model from learning other features.
*   **How to use the AI Splitter Agent:**
    *   **Instruction:** "Isolate the gradient flow. Instrument the training loop to hook into the backward pass. For each step, dump the L2 norm and standard deviation of the gradients for each major model component (embedding layer, attention blocks, LM_Head). We need to see if gradients are vanishing for most of the network or exploding in the final layer."
    *   **Agent's Action:** The agent would help modify the C++ and CUDA training code to simply  , run a short training session, and generate a report or plot to visualize where the gradient flow is breaking down.

**Scenario 2: Verifying a Custom CUDA Kernel Against a Reference**

*   **Problem:** You suspect a bug in a custom CUDA kernel (e.g., a fused attention or normalization layer) is silently corrupting the backward pass, leading to the training instability.
*   **How to use the AI Splitter Agent:**
    *   **Instruction:** "Create a numerical verification test for the custom RMSNorm CUDA kernel. The test should compare its forward and backward pass outputs to a simple, reference implementation from a library like PyTorch. Feed identical input tensors (including random, zero, and large values) to both implementations and assert that the outputs and resulting gradients are numerically identical within a small tolerance."
    *   **Agent's Action:** The agent would generate the necessary C++ or Python test harness code. This script would set up this "golden reference" test, execute it, and report any discrepancies, confirming or denying the kernel's correctness.

**Scenario 3: Analyzing Activation Distributions**

*   **Problem:** The training loss is stagnant. You're worried that neuron activations are saturating (e.g., all becoming very large or very small), which would effectively kill gradient flow.
*   **How to use the AI Splitter Agent:**
    *   **Instruction:** "Split the forward pass. Add hooks to one of the transformer blocks to capture the activation values after the self-attention and after the MLP's non-linearity (e.g., SwiGLU). Run a single forward pass with a typical batch of data. Then, generate histograms of these activation distributions. We need to check if they are well-distributed or if they are all clustered near zero or a saturation point."
    *   **Agent's Action:** The agent would assist in instrumenting the model's forward pass to dump these internal tensors and would then generate scripts to plot their distributions, helping to diagnose issues with the network's internal dynamic range.
