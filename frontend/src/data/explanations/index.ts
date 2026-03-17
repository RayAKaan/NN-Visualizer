export interface Explanation {
  icon: string;
  title: string;
  analogy?: string;
  paragraphs: string[];
  keyTerms?: Array<{ term: string; definition: string }>;
}

function dense(level: "simple" | "technical" | "mathematical"): Explanation {
  if (level === "mathematical") {
    return {
      icon: "??",
      title: "Dense Layer",
      paragraphs: [
        "Computes z = Wx + b where each output neuron is a weighted sum of all inputs.",
        "Backprop uses ?L/?W = dx^T and ?L/?x = W^Td.",
      ],
    };
  }
  if (level === "technical") {
    return {
      icon: "??",
      title: "Fully Connected Transformation",
      paragraphs: [
        "Every input feature connects to every output neuron through learned weights.",
        "This layer mixes features into a new representation used by later layers.",
      ],
      keyTerms: [
        { term: "W", definition: "weight matrix" },
        { term: "b", definition: "bias vector" },
      ],
    };
  }
  return {
    icon: "??",
    title: "Every Input Votes",
    analogy: "A committee vote where each input has a different influence.",
    paragraphs: [
      "Each output number is built from all input numbers plus a bias.",
      "Strong positive weights increase an output, negative weights suppress it.",
    ],
  };
}

function relu(level: "simple" | "technical" | "mathematical"): Explanation {
  if (level === "mathematical") {
    return {
      icon: "?",
      title: "ReLU",
      paragraphs: ["f(x)=max(0,x)", "Derivative is 1 for x>0 and 0 for x=0."],
    };
  }
  if (level === "technical") {
    return {
      icon: "?",
      title: "Non-Linearity Gate",
      paragraphs: [
        "Negative pre-activations are clamped to zero.",
        "This introduces non-linearity and sparsity.",
      ],
    };
  }
  return {
    icon: "?",
    title: "The Gatekeeper",
    analogy: "Positive values pass. Negative values are blocked.",
    paragraphs: ["ReLU keeps useful signals and drops weak/negative ones."],
  };
}

function conv(level: "simple" | "technical" | "mathematical"): Explanation {
  if (level === "mathematical") {
    return {
      icon: "??",
      title: "Conv2D",
      paragraphs: [
        "Each output pixel is a local dot product between kernel and input patch.",
        "Weight sharing makes the same detector scan every location.",
      ],
    };
  }
  if (level === "technical") {
    return {
      icon: "??",
      title: "Feature Extractor",
      paragraphs: [
        "Kernels detect local patterns like edges and textures.",
        "Each filter generates one feature map.",
      ],
    };
  }
  return {
    icon: "??",
    title: "Pattern Scanner",
    analogy: "A small stencil sliding over the image.",
    paragraphs: ["High response means the filter found its pattern there."],
  };
}

function pool(level: "simple" | "technical" | "mathematical"): Explanation {
  if (level === "mathematical") {
    return { icon: "???", title: "Pooling", paragraphs: ["Takes local max to downsample spatial resolution."] };
  }
  if (level === "technical") {
    return { icon: "???", title: "Spatial Downsampling", paragraphs: ["Keeps strongest local activation and reduces compute."] };
  }
  return { icon: "???", title: "Keep the Strongest", analogy: "Only the strongest signal in each region survives.", paragraphs: ["This keeps key features and shrinks the map."] };
}

function lstm(level: "simple" | "technical" | "mathematical"): Explanation {
  if (level === "mathematical") {
    return {
      icon: "??",
      title: "LSTM Gates",
      paragraphs: ["f_t,i_t,o_t gates control memory update and output.", "C_t = f_t?C_{t-1}+i_t?C~_t."],
    };
  }
  if (level === "technical") {
    return {
      icon: "??",
      title: "Temporal Memory Cell",
      paragraphs: ["LSTM keeps long-term context while integrating current timestep input."],
    };
  }
  return {
    icon: "??",
    title: "Memory Notebook",
    analogy: "Remember useful parts, forget the rest, output a summary.",
    paragraphs: ["Each timestep updates memory before passing to the next."],
  };
}

export function getExplanation(
  layerType: string,
  _architecture: string,
  _dataset: string,
  level: "simple" | "technical" | "mathematical",
): Explanation {
  if (layerType === "dense") return dense(level);
  if (layerType === "activation_relu") return relu(level);
  if (layerType === "conv2d") return conv(level);
  if (layerType === "max_pool") return pool(level);
  if (layerType === "lstm_cell") return lstm(level);
  if (layerType === "softmax") {
    return {
      icon: "??",
      title: "Softmax",
      paragraphs: ["Converts logits into probabilities that sum to 1."],
    };
  }
  if (layerType === "flatten") {
    return {
      icon: "??",
      title: "Flatten",
      paragraphs: ["Reshapes spatial maps into a 1D vector for dense layers."],
    };
  }
  if (layerType === "preprocessing") {
    return {
      icon: "??",
      title: "Preprocessing",
      paragraphs: ["Normalizes and prepares input values for stable network behavior."],
    };
  }
  if (layerType === "input") {
    return {
      icon: "???",
      title: "Input",
      paragraphs: ["Raw pixel values enter the network pipeline."],
    };
  }
  return { icon: "??", title: layerType, paragraphs: ["This stage transforms data for the next stage."] };
}
