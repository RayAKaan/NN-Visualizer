import React from "react";
import { CNNPredictionResult, PredictionResult } from "../../types";

interface Props {
  annResult: PredictionResult | null;
  cnnResult: CNNPredictionResult | null;
}

const ComparisonView: React.FC<Props> = ({ annResult, cnnResult }) => {
  const agree = annResult && cnnResult ? annResult.prediction === cnnResult.prediction : false;

  return (
    <div className="comparison-container">
      <div className="comparison-panel">
        <div className="comparison-panel-header"><span className="model-badge ann">ANN</span></div>
        {annResult ? (
          <div>
            <p>Prediction: {annResult.prediction}</p>
            <p>Confidence: {(Math.max(...annResult.probabilities) * 100).toFixed(1)}%</p>
          </div>
        ) : <p>No ANN result yet.</p>}
      </div>
      <div className="comparison-panel">
        <div className="comparison-panel-header"><span className="model-badge cnn">CNN</span></div>
        {cnnResult ? (
          <div>
            <p>Prediction: {cnnResult.prediction}</p>
            <p>Confidence: {(cnnResult.confidence * 100).toFixed(1)}%</p>
          </div>
        ) : <p>No CNN result yet.</p>}
      </div>
      <div className="comparison-summary">
        {annResult && cnnResult ? (
          <>
            <p className={agree ? "comparison-agree" : "comparison-disagree"}>
              {agree ? `Both models agree: digit ${annResult.prediction}` : `Models disagree: ANN=${annResult.prediction}, CNN=${cnnResult.prediction}`}
            </p>
            <p>
              Confidence delta: {Math.abs((Math.max(...annResult.probabilities) - cnnResult.confidence) * 100).toFixed(1)}%
            </p>
          </>
        ) : <p>Draw a digit to compare ANN vs CNN.</p>}
      </div>
    </div>
  );
};

export default ComparisonView;
