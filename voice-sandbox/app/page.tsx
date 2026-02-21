"use client";

import { useState, useRef, useEffect, useCallback } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type Screen = "role-select" | "voice-session" | "job-confirmed";
type Role = "CLIENT" | "WORKER";
type Lang = "hi" | "mr" | "gu" | "en";

interface CollectedParams {
  job_type?: string;
  location?: string;
  date?: string;
  time?: string;
  budget?: string;
  description?: string;
}

interface ChatMessage {
  role: "User" | "AI" | "System";
  text: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const LANG_LABELS: Record<Lang, string> = {
  hi: "Hindi",
  mr: "Marathi",
  gu: "Gujarati",
  en: "English",
};

const PARAM_LABELS: Record<string, string> = {
  job_type: "Service",
  location: "Location",
  date: "Date",
  time: "Time",
  budget: "Budget",
  description: "Details",
};

const GREETINGS: Record<Lang, string> = {
  hi: "Namaste! Aapko kaunsi seva chahiye?",
  mr: "Namaskar! Tumhala kashi seva pahije?",
  gu: "Kem cho! Tamane kai seva joie che?",
  en: "Hello! How may I help you today?",
};

const API_URL = process.env.NEXT_PUBLIC_API_URL;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Plays a TTS audio blob via /api/tts and returns a Promise that resolves
 * when the audio finishes playing.
 * Exported cleanly so the main app can import this as a utility.
 */
async function speakText(text: string, lang: Lang): Promise<void> {
  const res = await fetch(`${API_URL}/api/tts`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify({ text, lang }),
  });
  const data = await res.json();
  if (!data.audio_b64) return;
  return new Promise((resolve) => {
    const audio = new Audio(`data:audio/mp3;base64,${data.audio_b64}`);
    audio.onended = () => resolve();
    audio.onerror = () => resolve();
    // play() returns a Promise — must catch rejection or browser throws an
    // unhandled error when autoplay is blocked.
    audio.play().catch(() => resolve());
  });
}

/**
 * Records audio from the microphone until stopRecording() is called.
 * Returns raw Blob chunks as a single WebM blob.
 * Exported cleanly for reuse in the main app.
 */
function useAudioRecorder() {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async (): Promise<void> => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunksRef.current = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.start();
    mediaRecorderRef.current = recorder;
  }, []);

  const stopRecording = useCallback((): Promise<Blob> => {
    return new Promise((resolve) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder) {
        resolve(new Blob([], { type: "audio/webm" }));
        return;
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        // Kill mic indicator light
        recorder.stream.getTracks().forEach((t) => t.stop());
        resolve(blob);
      };
      recorder.stop();
    });
  }, []);

  return { startRecording, stopRecording };
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------
export default function VoiceSandbox() {
  const [screen, setScreen] = useState<Screen>("role-select");
  const [role, setRole] = useState<Role>("CLIENT");
  const [lang, setLang] = useState<Lang>("hi");

  // Voice session state
  const [collectedParams, setCollectedParams] = useState<CollectedParams>({});
  const [isConfirmationStep, setIsConfirmationStep] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [sessionStarted, setSessionStarted] = useState(false); // waiting for first tap
  const [status, setStatus] = useState("");
  const [chatLog, setChatLog] = useState<ChatMessage[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const { startRecording, stopRecording } = useAudioRecorder();

  // Auto-scroll chat log
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatLog]);

  // ---------------------------------------------------------------------------
  // AI greeting — called DIRECTLY from a button click so the browser allows
  // audio autoplay (browsers block audio not tied to a real user gesture).
  // ---------------------------------------------------------------------------
  const hasGreeted = useRef(false);
  const handleStartSession = useCallback(async () => {
    if (hasGreeted.current) return;
    hasGreeted.current = true;
    setSessionStarted(true);
    setIsBusy(true);
    setStatus("AI is speaking...");
    const greeting = GREETINGS[lang];
    setChatLog([{ role: "AI", text: greeting }]);
    await speakText(greeting, lang);
    setIsBusy(false);
    setStatus("Hold the button and speak.");
  }, [lang]);

  // ---------------------------------------------------------------------------
  // Core voice turn
  // ---------------------------------------------------------------------------
  const handleMicPress = useCallback(async () => {
    if (isBusy) return;
    try {
      await startRecording();
      setIsRecording(true);
      setStatus("Listening...");
    } catch {
      setStatus("Error: Microphone access denied.");
    }
  }, [isBusy, startRecording]);

  const handleMicRelease = useCallback(async () => {
    if (!isRecording) return;
    setIsRecording(false);
    setIsBusy(true);

    try {
      // 1. Get audio blob
      const audioBlob = await stopRecording();

      // 2. STT
      setStatus("Transcribing...");
      const formData = new FormData();
      formData.append("audio", audioBlob, "audio.webm");
      const sttRes = await fetch(`${API_URL}/api/stt`, {
        method: "POST",
        headers: { "ngrok-skip-browser-warning": "true" },
        body: formData,
      });
      if (!sttRes.ok) throw new Error(`STT failed (${sttRes.status})`);
      const sttData = await sttRes.json();
      const userText = String(sttData.transcript || "").trim();
      if (!userText) {
        setStatus("Could not understand. Please try again.");
        setIsBusy(false);
        return;
      }
      setChatLog((prev) => [...prev, { role: "User", text: userText }]);

      // 3. Voice session (multi-turn brain)
      setStatus("Thinking...");
      const sessionRes = await fetch(`${API_URL}/api/voice_session`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({
          transcript: userText,
          collected_params: collectedParams,
          language: lang,
          is_confirmation: isConfirmationStep,
        }),
      });
      if (!sessionRes.ok) {
        const detail = await sessionRes.text();
        throw new Error(`Session failed (${sessionRes.status}): ${detail}`);
      }
      const sessionData = await sessionRes.json();

      // Update collected params
      setCollectedParams(sessionData.collected_params);

      const aiReply: string = sessionData.next_question || "";
      setChatLog((prev) => [...prev, { role: "AI", text: aiReply }]);

      // Handle state transitions
      if (sessionData.job_confirmed) {
        // Job is confirmed — play final message and go to confirmation screen
        setStatus("Job confirmed! ✅");
        await speakText(aiReply, lang);
        setScreen("job-confirmed");
        setIsBusy(false);
        return;
      }

      if (sessionData.is_complete && !isConfirmationStep) {
        // All params collected — next user turn is a confirmation
        setIsConfirmationStep(true);
      }

      // 4. TTS — play the AI's next question
      setStatus("AI is speaking...");
      await speakText(aiReply, lang);
      setStatus("Hold the button and speak.");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`Error: ${msg}`);
      setChatLog((prev) => [...prev, { role: "System", text: `Error: ${msg}` }]);
    } finally {
      setIsBusy(false);
    }
  }, [isRecording, collectedParams, lang, isConfirmationStep, stopRecording]);

  // ---------------------------------------------------------------------------
  // Reset / Start again
  // ---------------------------------------------------------------------------
  const resetSession = useCallback(() => {
    hasGreeted.current = false;
    setCollectedParams({});
    setIsConfirmationStep(false);
    setSessionStarted(false);
    setChatLog([]);
    setStatus("");
    setScreen("voice-session");
  }, []);

  // ---------------------------------------------------------------------------
  // UI
  // ---------------------------------------------------------------------------

  // ── Screen 1: Role Select ──
  if (screen === "role-select") {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center gap-10 p-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight mb-2">KaamSetu</h1>
          <p className="text-gray-400 text-lg">Voice Assistant — Job Posting Demo</p>
        </div>

        {/* Role picker */}
        <div className="flex gap-6">
          {(["CLIENT", "WORKER"] as Role[]).map((r) => (
            <button
              key={r}
              onClick={() => setRole(r)}
              className={`w-44 h-44 rounded-2xl text-lg font-bold flex flex-col items-center justify-center gap-3 border-2 transition-all
                ${role === r
                  ? "bg-orange-500 border-orange-400 scale-105 shadow-lg shadow-orange-900"
                  : "bg-gray-800 border-gray-700 hover:border-gray-500"
                }`}
            >
              <span className="text-4xl">{r === "CLIENT" ? "🔧" : "💼"}</span>
              <span>{r === "CLIENT" ? "I Need a Worker" : "I Want Work"}</span>
            </button>
          ))}
        </div>

        {/* Language picker */}
        <div className="flex flex-col items-center gap-3">
          <p className="text-gray-400 text-sm uppercase tracking-widest">Select Language</p>
          <div className="flex gap-3">
            {(Object.keys(LANG_LABELS) as Lang[]).map((l) => (
              <button
                key={l}
                onClick={() => setLang(l)}
                className={`px-5 py-2 rounded-full text-sm font-semibold border transition-all
                  ${lang === l
                    ? "bg-blue-600 border-blue-400 text-white"
                    : "bg-gray-800 border-gray-600 text-gray-300 hover:border-gray-400"
                  }`}
              >
                {LANG_LABELS[l]}
              </button>
            ))}
          </div>
        </div>

        {/* Go button — only show for CLIENT (job posting is client-side) */}
        <button
          onClick={() => {
            if (role === "CLIENT") setScreen("voice-session");
            // WORKER flow placeholder — to be added in main app
            else setStatus("Worker flow coming soon!");
          }}
          className="mt-4 px-10 py-4 bg-orange-500 hover:bg-orange-400 text-white font-bold text-lg rounded-full shadow-lg transition-all active:scale-95"
        >
          {role === "CLIENT" ? "Post a Job →" : "Find Work →"}
        </button>
      </div>
    );
  }

  // ── Screen 3: Job Confirmed ──
  if (screen === "job-confirmed") {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center gap-8 p-8">
        <div className="text-7xl">✅</div>
        <h2 className="text-3xl font-bold">Job Posted!</h2>

        {/* Summary card */}
        <div className="bg-gray-900 rounded-2xl p-6 w-full max-w-md space-y-3">
          {Object.entries(collectedParams).map(([key, value]) =>
            value ? (
              <div key={key} className="flex justify-between text-sm">
                <span className="text-gray-400 capitalize">{PARAM_LABELS[key] ?? key}</span>
                <span className="text-white font-medium capitalize">{value}</span>
              </div>
            ) : null
          )}
        </div>

        <button
          onClick={() => {
            setScreen("role-select");
            hasGreeted.current = false;
            setCollectedParams({});
            setIsConfirmationStep(false);
            setChatLog([]);
          }}
          className="px-8 py-3 bg-gray-700 hover:bg-gray-600 rounded-full font-semibold transition-all"
        >
          ← Back to Start
        </button>
      </div>
    );
  }

  // ── Screen 2: Voice Session ──
  // Determine filled vs missing params for the live status panel
  const requiredParams = ["job_type", "location", "date", "time", "budget"] as const;
  const filledCount = requiredParams.filter((p) => collectedParams[p]).length;
  const progressPct = Math.round((filledCount / requiredParams.length) * 100);

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center gap-6 p-6 pt-10">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl font-bold">Post a Job</h1>
        <p className="text-gray-400 text-sm mt-1">{LANG_LABELS[lang]} · {role}</p>
      </div>

      {/* Progress bar */}
      <div className="w-full max-w-md">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Parameters collected</span>
          <span>{filledCount}/{requiredParams.length}</span>
        </div>
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-orange-500 rounded-full transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Collected params chips */}
      <div className="flex flex-wrap gap-2 w-full max-w-md justify-center">
        {requiredParams.map((p) => (
          <span
            key={p}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-all
              ${collectedParams[p]
                ? "bg-green-900 border-green-600 text-green-300"
                : "bg-gray-800 border-gray-700 text-gray-500"
              }`}
          >
            {PARAM_LABELS[p]}: {collectedParams[p] ?? "—"}
          </span>
        ))}
      </div>

      {/* Mic button / Start button */}
      <div className="flex flex-col items-center gap-4 mt-2">
        {!sessionStarted ? (
          // First interaction — must be a direct click to unlock browser audio
          <button
            onClick={handleStartSession}
            className="w-36 h-36 rounded-full text-base font-bold select-none transition-all bg-orange-500 hover:bg-orange-400 shadow-xl shadow-orange-900 active:scale-95 animate-pulse"
          >
            🔊 Tap to Start
          </button>
        ) : (
          <button
            onMouseDown={handleMicPress}
            onMouseUp={handleMicRelease}
            onTouchStart={(e) => { e.preventDefault(); handleMicPress(); }}
            onTouchEnd={(e) => { e.preventDefault(); handleMicRelease(); }}
            disabled={isBusy}
            className={`w-36 h-36 rounded-full text-lg font-bold select-none transition-all
              ${isRecording
                ? "bg-red-600 scale-110 shadow-2xl shadow-red-900 animate-pulse"
                : isBusy
                  ? "bg-gray-700 cursor-not-allowed opacity-60"
                  : "bg-blue-600 hover:bg-blue-500 shadow-xl shadow-blue-900 active:scale-95"
              }`}
          >
            {isRecording ? "🎙 Release" : isBusy ? "⏳" : "🎙 Hold"}
          </button>
        )}
        <p className="text-gray-400 text-sm text-center max-w-xs">{status}</p>
      </div>

      {/* Chat log */}
      <div className="w-full max-w-md flex-1 bg-gray-900 rounded-2xl p-4 font-mono text-sm overflow-y-auto max-h-64">
        {chatLog.length === 0 && (
          <p className="text-gray-600 text-center mt-8">Conversation will appear here...</p>
        )}
        {chatLog.map((msg, i) => (
          <div
            key={i}
            className={`mb-3 ${msg.role === "User"
              ? "text-blue-400"
              : msg.role === "AI"
                ? "text-green-400"
                : "text-red-400"
              }`}
          >
            <strong>{msg.role}: </strong>
            <span>{msg.text}</span>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Back / Reset */}
      <div className="flex gap-4">
        <button
          onClick={() => setScreen("role-select")}
          className="text-gray-500 hover:text-gray-300 text-sm transition-all"
        >
          ← Change Role
        </button>
        <button
          onClick={resetSession}
          className="text-gray-500 hover:text-gray-300 text-sm transition-all"
        >
          ↺ Start Over
        </button>
      </div>
    </div>
  );
}