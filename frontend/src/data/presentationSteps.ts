export const presentationSteps = [
  { id: "intro", title: "Welcome", description: "Interactive NN visualization", highlight: "topbar", autoAction: "", detail: "Explore ANN/CNN/RNN behavior." },
  { id: "canvas", title: "Canvas", description: "Draw a digit", highlight: "canvas", autoAction: "", detail: "Use mouse or touch." },
  { id: "input", title: "Input", description: "28x28 pixels", highlight: "left", autoAction: "", detail: "Pixels feed each model." },
  { id: "hidden", title: "Hidden Layers", description: "Dense/Conv/LSTM internals", highlight: "center", autoAction: "", detail: "Activation flow." },
  { id: "conn", title: "Connections", description: "Weights and edges", highlight: "center", autoAction: "", detail: "ANN edge strengths." },
  { id: "output", title: "Output", description: "Digit probabilities", highlight: "right", autoAction: "", detail: "Top class wins." },
  { id: "prediction", title: "Prediction", description: "Confidence diagnostics", highlight: "prediction", autoAction: "", detail: "Model explanation panel." },
  { id: "3d", title: "3D View", description: "Spatial network structure", highlight: "center", autoAction: "switchTo3D", detail: "ANN 3D scene." },
  { id: "cnn", title: "CNN", description: "Feature maps and kernels", highlight: "center", autoAction: "switchToCNN", detail: "Conv filter responses." },
  { id: "rnn", title: "RNN", description: "Sequential row processing", highlight: "center", autoAction: "switchToRNN", detail: "Timestep importance." },
  { id: "compare", title: "Compare", description: "Side-by-side models", highlight: "center", autoAction: "switchToCompare", detail: "Agreement/disagreement." },
  { id: "training", title: "Training", description: "Live metrics", highlight: "center", autoAction: "switchToTraining", detail: "Batch/epoch insights." },
  { id: "done", title: "Conclusion", description: "You are ready", highlight: "topbar", autoAction: "switchToPredict", detail: "Try your own digits." }
];
