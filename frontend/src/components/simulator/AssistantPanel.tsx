import React, { useState } from "react";
import { useAssistantStore } from "../../store/assistantStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";
import { NeuralInput } from "@/design-system/components/NeuralInput";

export function AssistantPanel() {
  const [input, setInput] = useState("");
  const { open, toggle, send, loading, messages } = useAssistantStore();

  return (
    <div className="assistant-wrap">
      <NeuralButton variant="primary" onClick={toggle} className="assistant-toggle">
        {open ? "Close AI" : "Open AI"}
      </NeuralButton>
      {open && (
        <NeuralPanel className="assistant-panel" variant="elevated">
          <div className="assistant-title">Assistant</div>
          <div className="assistant-messages neural-scroll-area">
            {messages.map((m, idx) => (
              <div key={idx} className={`assistant-message ${m.role === "user" ? "user" : "ai"}`}>
                <strong>{m.role === "user" ? "You" : "AI"}:</strong> {m.text}
              </div>
            ))}
            {messages.length === 0 && <div className="assistant-empty">Ask me to build an architecture.</div>}
          </div>
          <div className="assistant-input">
            <NeuralInput value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a question" />
            <NeuralButton
              variant="primary"
              disabled={!input || loading}
              onClick={() => {
                const text = input.trim();
                if (!text) return;
                void send(text);
                setInput("");
              }}
            >
              Send
            </NeuralButton>
          </div>
        </NeuralPanel>
      )}
    </div>
  );
}
