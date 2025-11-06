import React, { useState, useRef, useEffect } from 'react';
import { analyzeStudentFrame, ai } from '../services/geminiService';
import { ProctoringResult } from '../types';
import { LiveServerMessage, Modality } from '@google/genai';
import { createBlob } from '../services/audioUtils';

interface LobbyProps {
  onStartExam: () => void;
}

const ProctoringStatus: React.FC<{ result: ProctoringResult }> = ({ result }) => {
  const isClear = result.reason === "All clear";
  const isInitial = result.reason === "Initializing..." || result.reason === 'Starting proctoring...';
  const isError = result.reason.includes("AI analysis failed") || result.reason.includes("Could not access camera");

  let color = 'text-amber-400 border-amber-500/50 bg-amber-900/20';
  let icon = <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />;

  if (isClear) {
    color = 'text-green-400 border-green-500/50 bg-green-900/20';
    icon = <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />;
  } else if (isError) {
    color = 'text-red-400 border-red-500/50 bg-red-900/20';
  } else if (isInitial) {
    color = 'text-gray-400 border-gray-700 bg-gray-900/20';
  }

  return (
    <div className={`flex items-center justify-center gap-3 p-3 rounded-lg border ${color}`}>
      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        {icon}
      </svg>
      <span className="font-semibold">{result.reason}</span>
    </div>
  );
};


const Lobby: React.FC<LobbyProps> = ({ onStartExam }) => {
  const [proctoringResult, setProctoringResult] = useState<ProctoringResult>({ isCheating: false, reason: 'Initializing...' });
  const [isSetupActive, setIsSetupActive] = useState(false);
  const [proctoringPassed, setProctoringPassed] = useState(false);
  const [clearChecks, setClearChecks] = useState(0);

  const [micCheckState, setMicCheckState] = useState<'idle' | 'starting' | 'checking' | 'passed' | 'error'>('idle');
  const [micTranscript, setMicTranscript] = useState('');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const proctoringIntervalRef = useRef<number | null>(null);

  const micSessionPromiseRef = useRef<Promise<any> | null>(null);
  const micAudioContextRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const micScriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const micMediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const stopMicCheck = () => {
    micSessionPromiseRef.current?.then(session => session.close());
    micScriptProcessorRef.current?.disconnect();
    micMediaStreamSourceRef.current?.disconnect();
    micStreamRef.current?.getTracks().forEach(track => track.stop());
    if (micAudioContextRef.current && micAudioContextRef.current.state !== 'closed') {
        micAudioContextRef.current.close();
    }
  };

  useEffect(() => {
    return () => {
      if (proctoringIntervalRef.current) clearInterval(proctoringIntervalRef.current);
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
      stopMicCheck();
    };
  }, []);

  const startSetup = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
      setIsSetupActive(true);
      setProctoringResult({isCheating: false, reason: 'Starting proctoring...'});
      
      proctoringIntervalRef.current = window.setInterval(runProctoringCheck, 3000); // Faster check

    } catch (err) {
      console.error("Error accessing camera:", err);
      setProctoringResult({ isCheating: true, reason: 'Could not access camera. Please grant permission.' });
    }
  };

  const runProctoringCheck = async () => {
    if (!videoRef.current || !canvasRef.current || videoRef.current.readyState < 2) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    if (context) {
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64ImageData = canvas.toDataURL('image/jpeg').split(',')[1];
      
      const result = await analyzeStudentFrame(base64ImageData);
      setProctoringResult(result);
      
      if (!result.isCheating && result.reason === "All clear") {
        setClearChecks(prev => {
          const newCount = prev + 1;
          if (newCount >= 2) { // Require 2 consecutive clear checks
            setProctoringPassed(true);
            if (proctoringIntervalRef.current) clearInterval(proctoringIntervalRef.current);
          }
          return newCount;
        });
      } else {
        setClearChecks(0); // Reset count if a check fails
      }
    }
  };

  const handleStartMicCheck = async () => {
    setMicCheckState('starting');
    setMicTranscript('');
    try {
        micStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        micAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
        const audioContext = micAudioContextRef.current;
        
        micMediaStreamSourceRef.current = audioContext.createMediaStreamSource(micStreamRef.current);
        const source = micMediaStreamSourceRef.current;

        const scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        micScriptProcessorRef.current = scriptProcessor;

        micSessionPromiseRef.current = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            config: {
                responseModalities: [Modality.AUDIO], // This was missing
                inputAudioTranscription: {},
            },
            callbacks: {
                onopen: () => setMicCheckState('checking'),
                onmessage: (message: LiveServerMessage) => {
                    if (message.serverContent?.inputTranscription) {
                         const newText = message.serverContent.inputTranscription.text;
                        if(newText) setMicTranscript(prev => (prev + ' ' + newText).trim());
                    }
                },
                onerror: (e) => {
                    console.error("Mic check connection error:", e);
                    setMicCheckState('error');
                },
                onclose: () => {
                   if (micCheckState !== 'passed') setMicCheckState('idle');
                }
            }
        });
        
        scriptProcessor.onaudioprocess = (event) => {
            const inputData = event.inputBuffer.getChannelData(0);
            const pcmBlob = createBlob(inputData);
            micSessionPromiseRef.current?.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
            });
        };

        const gainNode = audioContext.createGain();
        gainNode.gain.setValueAtTime(0, audioContext.currentTime);

        source.connect(scriptProcessor);
        scriptProcessor.connect(gainNode);
        gainNode.connect(audioContext.destination);

    } catch (err) {
        console.error("Error setting up mic check:", err);
        setMicCheckState('error');
    }
  };
  
  const handleConfirmMic = () => {
    setMicCheckState('passed');
    stopMicCheck();
    onStartExam();
  };

  const renderActionArea = () => {
    if (!isSetupActive) {
        return (
            <button onClick={startSetup} className="w-full px-4 py-3 text-lg font-semibold text-white bg-violet-700 rounded-md hover:bg-violet-800 transition-all transform hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-violet-500 focus:ring-offset-black">
              Start Setup
            </button>
        );
    }

    if (!proctoringPassed) {
        return (
             <button disabled className="w-full px-4 py-3 text-lg font-semibold text-white bg-gray-600 rounded-md cursor-not-allowed opacity-70">
              Proctoring in Progress...
            </button>
        );
    }
    
    // Microphone Check UI
    return (
        <div className="w-full p-4 space-y-3 bg-gray-800/50 rounded-lg border border-gray-700/50 transition-all">
            <h3 className="text-lg font-semibold text-center text-cyan-300">
                Step 2: Microphone Check
            </h3>
            {micCheckState === 'idle' && (
                <>
                    <p className="text-center text-gray-400 text-sm">Finally, let's make sure we can hear you.</p>
                    <button onClick={handleStartMicCheck} className="w-full px-4 py-2 font-semibold text-white bg-gradient-to-r from-cyan-600 to-blue-600 rounded-md hover:from-cyan-700 hover:to-blue-700 transition-all">
                        Start Mic Check
                    </button>
                </>
            )}
            {(micCheckState === 'starting' || micCheckState === 'checking') && (
                 <>
                    <p className="text-center text-gray-400 text-sm animate-pulse">Please say a few words, like "Testing one, two, three."</p>
                    <div className="min-h-[60px] w-full p-3 bg-black/40 rounded-md border border-gray-600 text-gray-200 italic transition-all">
                        {micTranscript || (micCheckState === 'starting' ? 'Initializing...' : 'Listening...')}
                    </div>
                    <button onClick={handleConfirmMic} disabled={!micTranscript} className="w-full px-4 py-2 font-semibold text-gray-900 bg-green-500 rounded-md transition-all disabled:bg-gray-600 disabled:text-white disabled:cursor-not-allowed hover:enabled:bg-green-600">
                        Confirm Mic is Working & Start Exam
                    </button>
                </>
            )}
             {micCheckState === 'error' && (
                <>
                    <p className="text-center text-red-400 text-sm">Could not start microphone. Please check browser permissions and try again.</p>
                     <button onClick={handleStartMicCheck} className="w-full px-4 py-2 font-semibold text-white bg-gradient-to-r from-amber-600 to-orange-600 rounded-md hover:from-amber-700 hover:to-orange-700 transition-all">
                        Retry Mic Check
                    </button>
                </>
            )}
            {micCheckState === 'passed' && (
                <div className="text-center text-green-400 animate-pulse font-semibold">
                    Starting Exam...
                </div>
            )}
        </div>
    );
  };


  return (
    <div className="flex items-center justify-center h-[calc(100vh-68px)] p-4">
      <div className="w-full max-w-2xl p-8 space-y-6 bg-black rounded-2xl shadow-2xl border border-gray-800">
        <div>
          <h2 className="text-3xl font-extrabold text-center text-white">
            Exam Setup & Integrity Check
          </h2>
          <p className="mt-2 text-center text-gray-400">
            Please complete the following steps to begin your exam.
          </p>
        </div>
        
        <div className="w-full aspect-video bg-black rounded-lg overflow-hidden border border-gray-700 relative">
          <video ref={videoRef} autoPlay muted className={`w-full h-full object-cover transform -scale-x-100 transition-opacity duration-500 ${isSetupActive ? 'opacity-100' : 'opacity-0'}`} />
          {!isSetupActive && (
             <div className="absolute inset-0 w-full h-full flex flex-col items-center justify-center text-white">
                 <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <p className="mt-2">Camera preview will appear here</p>
             </div>
          )}
        </div>
        <canvas ref={canvasRef} className="hidden"></canvas>

        {isSetupActive && <ProctoringStatus result={proctoringResult} />}
        
        <div className="pt-2">
            {renderActionArea()}
        </div>
      </div>
    </div>
  );
};

export default Lobby;