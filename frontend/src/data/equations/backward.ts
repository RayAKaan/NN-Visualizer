import type { BackwardEquationSet, StageDefinition } from "../../types/pipeline";

export function getBackwardEquation(stage: StageDefinition): BackwardEquationSet {
  switch (stage.type) {
    case "softmax":
      return {
        chainRule: "\\frac{\\partial L}{\\partial z_i} = \\hat{y}_i - y_i",
        localGradient: "\\frac{\\partial softmax_i}{\\partial z_j} = \\hat{y}_i(\\delta_{ij}-\\hat{y}_j)",
        explanation: "Softmax with cross-entropy yields prediction minus truth as the main error signal.",
      };
    case "dense":
      return {
        chainRule: "\\frac{\\partial L}{\\partial z} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z}",
        localGradient: "\\frac{\\partial z}{\\partial W} = a_{prev}^T",
        weightGradient: "\\frac{\\partial L}{\\partial W} = a_{prev}^T \\cdot \\delta",
        explanation: "Dense weight updates scale with both upstream error and previous activation strength.",
      };
    case "activation_relu":
      return {
        chainRule: "\\frac{\\partial L}{\\partial z} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z}",
        localGradient: "ReLU'(z)=1\\ (z>0),\\ 0\\ (z\\le0)",
        explanation: "ReLU blocks gradients for inactive neurons and passes gradients for active neurons.",
      };
    case "conv2d":
      return {
        chainRule: "\\frac{\\partial L}{\\partial K} = \\sum_{i,j} \\delta_{i,j} \\cdot X_{patch(i,j)}",
        localGradient: "\\frac{\\partial FM}{\\partial K}=X_{patch}",
        weightGradient: "\\Delta K=-\\eta\\frac{\\partial L}{\\partial K}",
        explanation: "Convolution kernels are updated by correlating error maps with local input patches.",
      };
    case "max_pool":
      return {
        chainRule: "\\frac{\\partial L}{\\partial x}=\\frac{\\partial L}{\\partial y}\\cdot\\frac{\\partial y}{\\partial x}",
        localGradient: "Gradient routes only to max elements of each pooling window",
        explanation: "MaxPool backward sends each gradient only through its winning location.",
      };
    case "flatten":
      return {
        chainRule: "\\frac{\\partial L}{\\partial X_{3D}} = reshape(\\frac{\\partial L}{\\partial x_{1D}})",
        localGradient: "reshape",
        explanation: "Flatten has no parameters; backward pass only reshapes gradient tensor.",
      };
    case "lstm_cell":
      return {
        chainRule: "BPTT: \\frac{\\partial L}{\\partial W} = \\sum_t \\frac{\\partial L_t}{\\partial W}",
        localGradient: "\\frac{\\partial C_t}{\\partial C_{t-1}} = f_t",
        explanation: "LSTM gates control how much gradient survives across time steps.",
      };
    default:
      return {
        chainRule: "\\frac{\\partial L}{\\partial in}=\\frac{\\partial L}{\\partial out}\\cdot\\frac{\\partial out}{\\partial in}",
        localGradient: "local derivative",
        explanation: "Chain rule pushes error signal toward earlier layers.",
      };
  }
}
