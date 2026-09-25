# Tensor Computation
### This directory focuses on naive, and optimized implementations of core language model computations name: softmax, and attention. It progresses from direct math-to-code formulation to memory efficient computation, as depicted below:
``Naive computation -> Numerically stable computation -> Online / incremental computation -> Tiled computation -> Memory efficient attention``

### The focus is on exploring how mathematically equivalent formulations when applied incrementally with controlled numeric range improves numerical stability and memory efficiency.