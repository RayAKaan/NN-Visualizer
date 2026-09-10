export const WORKSPACE_EVENTS = {
  predict: "neurofluxion:predict",
  resetLab: "neurofluxion:reset-lab",
  runForward: "neurofluxion:run-forward",
  startTraining: "neurofluxion:start-training",
  refreshModels: "neurofluxion:refresh-models",
  selectModel: "neurofluxion:select-model",
} as const;

export function emitWorkspaceEvent(name: string) {
  window.dispatchEvent(new Event(name));
}
