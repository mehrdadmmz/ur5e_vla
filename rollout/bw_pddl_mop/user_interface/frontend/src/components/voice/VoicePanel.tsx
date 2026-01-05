import { useState, useRef, useCallback } from 'react';

export function VoicePanel() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState('');
  const [feedback, setFeedback] = useState<{ success: boolean; message: string } | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        setIsProcessing(true);
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        chunksRef.current = [];

        // Convert to base64
        const reader = new FileReader();
        reader.onloadend = async () => {
          const base64 = (reader.result as string).split(',')[1];

          try {
            const response = await fetch('/api/voice/command', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                audio_base64: base64,
                format: 'webm',
              }),
            });

            if (response.ok) {
              const data = await response.json();
              setTranscription(data.transcript || '');
              setFeedback({
                success: data.executed,
                message: data.message,
              });
            } else {
              setFeedback({
                success: false,
                message: 'Failed to process voice command',
              });
            }
          } catch (err) {
            setFeedback({
              success: false,
              message: 'Network error',
            });
          } finally {
            setIsProcessing(false);
          }
        };
        reader.readAsDataURL(audioBlob);

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
      setTranscription('');
      setFeedback(null);
    } catch (err) {
      setFeedback({
        success: false,
        message: 'Microphone access denied',
      });
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  return (
    <div className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Voice Control</h2>

      {/* Push-to-talk button */}
      <div className="flex justify-center mb-4">
        <button
          onMouseDown={startRecording}
          onMouseUp={stopRecording}
          onMouseLeave={stopRecording}
          onTouchStart={startRecording}
          onTouchEnd={stopRecording}
          disabled={isProcessing}
          className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
            isRecording
              ? 'bg-red-600 scale-110 shadow-lg shadow-red-600/50'
              : isProcessing
              ? 'bg-gray-600 cursor-wait'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isProcessing ? (
            <svg className="animate-spin h-8 w-8 text-white" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          ) : (
            <svg
              className="w-8 h-8 text-white"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
          )}
        </button>
      </div>

      <p className="text-center text-sm text-gray-500 mb-4">
        {isRecording ? 'Recording... Release to send' : 'Hold to speak'}
      </p>

      {/* Transcription */}
      {transcription && (
        <div className="mb-3 p-3 bg-gray-100 rounded-lg">
          <div className="text-xs text-gray-500 mb-1">Heard:</div>
          <div className="text-gray-800">{transcription}</div>
        </div>
      )}

      {/* Feedback */}
      {feedback && (
        <div
          className={`p-3 rounded-lg ${
            feedback.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}
        >
          {feedback.message}
        </div>
      )}

      {/* Voice commands help */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="text-xs text-gray-500">
          <div className="font-medium text-gray-600 mb-1">Voice commands:</div>
          <div>"pause" / "continue" / "stop"</div>
          <div>"build bridge" / "build tower"</div>
          <div>"pick block 0" / "put down"</div>
        </div>
      </div>
    </div>
  );
}
