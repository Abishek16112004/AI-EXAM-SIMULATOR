import React, { useState, useRef, useEffect, useLayoutEffect, useCallback } from 'react';
import { LiveServerMessage, Modality } from '@google/genai';
import { streamDetailedExaminerFeedback, textToSpeech, analyzeStudentFrame, getExaminerQAPrompt, ai } from '../services/geminiService';
import { TranscriptEntry, ScoreData } from '../types';
import { decode, decodeAudioData, createBlob } from '../services/audioUtils';


const patientScript = `
You are a Standardized Patient in a clinical exam simulation named Alex. A medical student is interacting with you. Your goal is to accurately portray the patient so the student can practice their history-taking skills.

You have already greeted the student with "Hello, doctor."

**Patient Profile:**
- **Name:** Alex
- **Age:** 45
- **Occupation:** Office worker
- **Chief Complaint:** Persistent headaches.

**History of Present Illness (HPI):**
- The headaches started two weeks ago.
- The pain is a dull, throbbing sensation located behind the eyes.
- It gets worse in the afternoon, especially after looking at a computer screen for a long time.
- Over-the-counter pain relievers (like ibuprofen) provide only minimal relief.

**Your Behavior:**
- Your mood is slightly anxious because you're worried this might be serious.
- When the student responds to your greeting (e.g., "Hello Alex, what brings you in today?"), your next response should be to state your chief complaint. For example: "I've been having these persistent headaches."
- After stating your complaint, **crucially, do not volunteer information unless the student asks you directly.** 
- For example, if they ask 'when did they start?' or 'can you describe the pain?', only then provide more details.
- Keep your answers concise and conversational, based on the script.
- When the student concludes the interview (e.g., by saying 'Thank you, we're done for today'), respond politely (e.g., 'Thank you, doctor.') and do not offer further information.
`;

interface CallProps {
  onViolation: (reason: string) => void;
  isBlocked: boolean;
}

const Call: React.FC<CallProps> = ({ onViolation, isBlocked }) => {
    const [status, setStatus] = useState('Initializing...');
    const [examState, setExamState] = useState<'initializing' | 'ready' | 'in-progress' | 'feedback' | 'q&a' | 'ended'>('initializing');
    const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
    const [currentStudentTranscript, setCurrentStudentTranscript] = useState('');
    const [currentSpTranscript, setCurrentSpTranscript] = useState('');
    const [currentExaminerTranscript, setCurrentExaminerTranscript] = useState('');
    const [examinerFeedback, setExaminerFeedback] = useState('No feedback at this time.');
    const [scoreData, setScoreData] = useState<ScoreData | null>(null);
    const [isScoreVerified, setIsScoreVerified] = useState(false);
    const [isEditingScores, setIsEditingScores] = useState(false);
    const [editedScoreData, setEditedScoreData] = useState<ScoreData | null>(null);
    const [isExaminerSpeaking, setIsExaminerSpeaking] = useState(false);
    const [unclearAudioWarning, setUnclearAudioWarning] = useState(false);
    const [isPatientSpeaking, setIsPatientSpeaking] = useState(false);
    const [silenceWarningVisible, setSilenceWarningVisible] = useState(false);

    const localVideoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const transcriptEndRef = useRef<HTMLDivElement>(null);
    const sessionPromiseRef = useRef<Promise<any> | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const reconnectionAttemptRef = useRef(0);

    const inputAudioContextRef = useRef<AudioContext | null>(null);
    const outputAudioContextRef = useRef<AudioContext | null>(null);
    const nextStartTimeRef = useRef(0);
    const audioSourcesRef = useRef(new Set<AudioBufferSourceNode>());
    const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const silenceTimerRef = useRef<number | null>(null);
    const warningTimerRef = useRef<number | null>(null);
    
    // Create refs to hold the latest state to avoid stale closures in callbacks
    const transcriptsRef = useRef(transcripts);
    useEffect(() => {
        transcriptsRef.current = transcripts;
    }, [transcripts]);

    const examStateRef = useRef(examState);
    useEffect(() => {
        examStateRef.current = examState;
    }, [examState]);

    useLayoutEffect(() => {
        transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [transcripts, currentStudentTranscript, currentSpTranscript, currentExaminerTranscript]);
    
    const playAudio = async (base64Audio: string, onEnded: () => void = () => {}) => {
        if (!outputAudioContextRef.current || !base64Audio) return;
        const audioContext = outputAudioContextRef.current;
        const audioBuffer = await decodeAudioData(decode(base64Audio), audioContext, 24000, 1);
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        
        return new Promise<void>(resolve => {
            source.onended = () => {
                onEnded();
                resolve();
            };
            source.start();
        });
    };

    const connectToExaminerQASession = useCallback((feedback: string) => {
        let currentInput = '';
        let currentOutput = '';

        const systemInstruction = getExaminerQAPrompt(transcriptsRef.current, feedback);
    
        const qaSessionPromise = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
                inputAudioTranscription: {},
                outputAudioTranscription: {},
                systemInstruction: systemInstruction,
            },
            callbacks: {
                onopen: () => {
                    setStatus('You may ask the examiner questions now.');
                    if (inputAudioContextRef.current && mediaStreamSourceRef.current) {
                        scriptProcessorRef.current?.disconnect();
        
                        const qaScriptProcessor = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
                        scriptProcessorRef.current = qaScriptProcessor;
        
                        qaScriptProcessor.onaudioprocess = (event) => {
                            if (isBlocked) return;
                            const inputData = event.inputBuffer.getChannelData(0);
                            const pcmBlob = createBlob(inputData);
                            qaSessionPromise.then((session) => {
                                session.sendRealtimeInput({ media: pcmBlob });
                            });
                        };

                        const gainNode = inputAudioContextRef.current.createGain();
                        gainNode.gain.setValueAtTime(0, inputAudioContextRef.current.currentTime);
                        mediaStreamSourceRef.current.connect(qaScriptProcessor);
                        qaScriptProcessor.connect(gainNode);
                        gainNode.connect(inputAudioContextRef.current.destination);
                    }
                },
                onmessage: async (message: LiveServerMessage) => {
                    if (message.serverContent?.inputTranscription) currentInput += message.serverContent.inputTranscription.text;
                    if (message.serverContent?.outputTranscription) currentOutput += message.serverContent.outputTranscription.text;
                    setCurrentStudentTranscript(currentInput);
                    setCurrentExaminerTranscript(currentOutput);
    
                    if (message.serverContent?.turnComplete) {
                        const newTranscripts: TranscriptEntry[] = [];
                        if (currentInput.trim()) newTranscripts.push({ role: 'Student', text: currentInput.trim() });
                        if (currentOutput.trim()) newTranscripts.push({ role: 'Examiner', text: currentOutput.trim() });
                        if (newTranscripts.length > 0) setTranscripts(prev => [...prev, ...newTranscripts]);
                        currentInput = '';
                        currentOutput = '';
                        setCurrentStudentTranscript('');
                        setCurrentExaminerTranscript('');
                    }
                    
                    const audioData = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                    if (audioData && outputAudioContextRef.current) {
                        setIsExaminerSpeaking(true);
                        const audioBuffer = await decodeAudioData(decode(audioData), outputAudioContextRef.current, 24000, 1);
                        const source = outputAudioContextRef.current.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(outputAudioContextRef.current.destination);
                        source.addEventListener('ended', () => setIsExaminerSpeaking(false));
                        source.start();
                    }
                },
                onerror: (e) => setStatus('Connection error during Q&A.'),
                onclose: () => {
                    if(examStateRef.current !== 'ended') handleEndExam();
                },
            },
        });
        sessionPromiseRef.current = qaSessionPromise;
    }, [isBlocked]);

    const triggerFeedbackPhase = useCallback(async () => {
        if (examStateRef.current !== 'in-progress') return;
    
        if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
        if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
        setSilenceWarningVisible(false);
    
        setExamState('feedback');
        setStatus('Interaction complete. Generating feedback and scores...');
        setExaminerFeedback(''); // Start with an empty string
        sessionPromiseRef.current?.then(session => session.close());
        scriptProcessorRef.current?.disconnect();
    
        setTranscripts(prev => [...prev, { role: 'Examiner', text: '...' }]);
    
        let fullFeedbackText = '';
        let textBuffer = '';
        const audioQueue: Promise<void>[] = [];
    
        const playQueuedAudio = async (textToSpeak: string) => {
            if (!textToSpeak.trim()) return;
            const audioData = await textToSpeech(textToSpeak, 'Kore');
            if (audioData) {
                const lastPromise = audioQueue.length > 0 ? audioQueue[audioQueue.length - 1] : Promise.resolve();
                const newPlaybackPromise = lastPromise.then(() => {
                    setIsExaminerSpeaking(true);
                    return playAudio(audioData);
                });
                audioQueue.push(newPlaybackPromise);
            }
        };
        
        const processTextForSpeech = (isLastChunk: boolean = false) => {
            const punctuation = ['.', '?', '!', '\n'];
            
            while (true) {
                const firstPunctuationIndex = textBuffer.split('').findIndex(char => punctuation.includes(char));
                if (firstPunctuationIndex === -1) break;
                const sentence = textBuffer.substring(0, firstPunctuationIndex + 1);
                textBuffer = textBuffer.substring(firstPunctuationIndex + 1);
                if (sentence.trim()) playQueuedAudio(sentence);
            }

            if (isLastChunk && textBuffer.trim()) {
                playQueuedAudio(textBuffer);
                textBuffer = '';
            }
        };
        
        const onChunkReceived = (textChunk: string) => {
            fullFeedbackText += textChunk;
            setExaminerFeedback(fullFeedbackText);
            setTranscripts(prev => {
                const newTranscripts = [...prev];
                if (newTranscripts.length > 0 && newTranscripts[newTranscripts.length - 1].role === 'Examiner') {
                    newTranscripts[newTranscripts.length - 1].text = fullFeedbackText;
                }
                return newTranscripts;
            });
    
            textBuffer += textChunk;
            processTextForSpeech();
        };
    
        const { feedbackText, scoreData } = await streamDetailedExaminerFeedback(transcriptsRef.current, onChunkReceived);
        
        setExaminerFeedback(feedbackText);
        setTranscripts(prev => {
            const newTranscripts = [...prev];
            if (newTranscripts.length > 0 && newTranscripts[newTranscripts.length - 1].role === 'Examiner') {
                newTranscripts[newTranscripts.length - 1].text = feedbackText;
            }
            return newTranscripts;
        });
        setScoreData(scoreData);

        processTextForSpeech(true);
    
        await Promise.all(audioQueue);
        
        setIsExaminerSpeaking(false);
        setExamState('q&a');
        connectToExaminerQASession(feedbackText);
    
    }, [connectToExaminerQASession]);

    const resetSilenceTimer = useCallback(() => {
        if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
        if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
        setSilenceWarningVisible(false);

        if (examStateRef.current === 'in-progress') { 
            warningTimerRef.current = window.setTimeout(() => {
                setSilenceWarningVisible(true);
            }, 15000); // 15 seconds to show warning

            silenceTimerRef.current = window.setTimeout(() => {
                setSilenceWarningVisible(false);
                triggerFeedbackPhase();
            }, 20000); // 20 seconds of silence triggers feedback
        }
    }, [triggerFeedbackPhase]);

    const runProctoringCheck = useCallback(async () => {
        if (!localVideoRef.current || !canvasRef.current || localVideoRef.current.readyState < 2 || isBlocked) return;
    
        const video = localVideoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        if (context) {
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          const base64ImageData = canvas.toDataURL('image/jpeg').split(',')[1];
          
          const result = await analyzeStudentFrame(base64ImageData);
          if (result.isCheating) {
            onViolation(result.reason);
          }
        }
    }, [isBlocked, onViolation]);

    useEffect(() => {
        if (examState === 'in-progress' && !isBlocked) {
            const intervalId = setInterval(runProctoringCheck, 7000);
            return () => clearInterval(intervalId);
        }
    }, [examState, isBlocked, runProctoringCheck]);

    const connectToPatientSession = () => {
        let currentInput = '';
        let currentOutput = '';
        
        sessionPromiseRef.current = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                inputAudioTranscription: {},
                outputAudioTranscription: {},
                systemInstruction: patientScript,
            },
            callbacks: {
                onopen: () => {
                    setStatus('Live session open.');
                    reconnectionAttemptRef.current = 0;
                },
                onmessage: async (message: LiveServerMessage) => {
                    if (message.serverContent?.inputTranscription) currentInput += message.serverContent.inputTranscription.text;
                    if (message.serverContent?.outputTranscription) currentOutput += message.serverContent.outputTranscription.text;
                    setCurrentStudentTranscript(currentInput);
                    setCurrentSpTranscript(currentOutput);

                    if (message.serverContent?.turnComplete) {
                        const studentText = currentInput.trim();
                        const spText = currentOutput.trim();

                        if (studentText.length === 0 && spText.length === 0 && currentInput.length > 0) {
                            setUnclearAudioWarning(true);
                            setTimeout(() => setUnclearAudioWarning(false), 3000);
                        }

                        const newTranscripts: TranscriptEntry[] = [];
                        if (studentText) newTranscripts.push({ role: 'Student', text: studentText });
                        if (spText) newTranscripts.push({ role: 'SP', text: spText });
                        
                        if(newTranscripts.length > 0) {
                             setTranscripts(prev => [...prev, ...newTranscripts]);
                        }

                        resetSilenceTimer(); // Reset silence timer after each turn.

                        currentInput = '';
                        currentOutput = '';
                        setCurrentStudentTranscript('');
                        setCurrentSpTranscript('');
                    }
                    
                    const audioData = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                    if (audioData && outputAudioContextRef.current) {
                        setIsPatientSpeaking(true);
                        nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputAudioContextRef.current.currentTime);
                        const audioBuffer = await decodeAudioData(decode(audioData), outputAudioContextRef.current, 24000, 1);
                        const source = outputAudioContextRef.current.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(outputAudioContextRef.current.destination);
                        source.addEventListener('ended', () => {
                            audioSourcesRef.current.delete(source);
                            if (audioSourcesRef.current.size === 0) setIsPatientSpeaking(false);
                        });
                        source.start(nextStartTimeRef.current);
                        nextStartTimeRef.current += audioBuffer.duration;
                        audioSourcesRef.current.add(source);
                    }
                },
                onerror: (e) => setStatus('Connection error. Attempting to reconnect...'),
                onclose: () => {
                    if (examStateRef.current === 'in-progress') {
                        setStatus('Connection closed. Reconnecting...');
                        if (reconnectionAttemptRef.current < 5) {
                            setTimeout(() => {
                                reconnectionAttemptRef.current++;
                                connectToPatientSession();
                            }, 2000 * reconnectionAttemptRef.current);
                        } else {
                            setStatus('Connection failed permanently. Please refresh.');
                        }
                    } else {
                        setStatus('Patient session closed.');
                    }
                },
            },
        });
    };

    useEffect(() => {
        const initializeExam = async () => {
            try {
                setStatus('Requesting permissions...');
                streamRef.current = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
                if (localVideoRef.current) localVideoRef.current.srcObject = streamRef.current;

                setStatus('Initializing Audio Context...');
                inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
                // FIX: Remove sampleRate hint to prevent playback speed issues in some browsers.
                outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
                
                if (inputAudioContextRef.current && streamRef.current) {
                    mediaStreamSourceRef.current = inputAudioContextRef.current.createMediaStreamSource(streamRef.current);
                }

                setStatus('Connecting to AI...');
                connectToPatientSession();

                setStatus('Generating introductions...');
                const examinerGreetingText = "Welcome to the clinical exam. I will be observing. The patient will begin once you are ready.";
                setExaminerFeedback(examinerGreetingText);
                const [spGreetingAudio, examinerGreetingAudio] = await Promise.all([
                    textToSpeech("Hello. Please press Ready when you wish to begin.", 'Zephyr'),
                    textToSpeech(examinerGreetingText, 'Kore')
                ]);
                
                if (spGreetingAudio) await playAudio(spGreetingAudio);
                if (examinerGreetingAudio) await playAudio(examinerGreetingAudio);

                setStatus('Please press Ready to begin.');
                setExamState('ready');

            } catch (err) {
                console.error("Error setting up session:", err);
                setStatus('Error: Could not access camera or microphone.');
            }
        };

        initializeExam();

        return () => {
            sessionPromiseRef.current?.then(session => session.close());
            if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
            if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
            scriptProcessorRef.current?.disconnect();
            mediaStreamSourceRef.current?.disconnect();
            streamRef.current?.getTracks().forEach(track => track.stop());
            if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') {
                inputAudioContextRef.current.close();
            }
            if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
                outputAudioContextRef.current.close();
            }
            audioSourcesRef.current.forEach(source => source.stop());
        };
    }, []);

    useEffect(() => {
        if(isBlocked) {
            handleEndExam();
            setStatus('Exam Terminated.');
        }
    }, [isBlocked]);

    const handleStartExam = async () => {
        if (!inputAudioContextRef.current || !mediaStreamSourceRef.current) return;
        
        setExamState('in-progress');
        setStatus('Connecting microphone...');
        
        scriptProcessorRef.current?.disconnect();
        const scriptProcessor = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
        scriptProcessorRef.current = scriptProcessor;

        scriptProcessor.onaudioprocess = (event) => {
            if (isBlocked) return;
            const inputData = event.inputBuffer.getChannelData(0);
            const pcmBlob = createBlob(inputData);
            sessionPromiseRef.current?.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
            });
        };
        
        const gainNode = inputAudioContextRef.current.createGain();
        gainNode.gain.setValueAtTime(0, inputAudioContextRef.current.currentTime);
        mediaStreamSourceRef.current.connect(scriptProcessor);
        scriptProcessor.connect(gainNode);
        gainNode.connect(inputAudioContextRef.current.destination);
        
        setStatus('Connected. You may begin speaking.');
        
        try {
            const greetingAudio = await textToSpeech("Hello, doctor.", 'Zephyr');
            if (greetingAudio) {
                setIsPatientSpeaking(true);
                // Start silence timer after the patient finishes greeting
                await playAudio(greetingAudio, () => {
                    setIsPatientSpeaking(false);
                    resetSilenceTimer();
                });
            } else {
                 resetSilenceTimer();
            }
        } catch (error) {
            console.error("Failed to generate SP greeting:", error);
            resetSilenceTimer();
        }
    };

    const handleEndExam = () => {
        setExamState('ended');
        sessionPromiseRef.current?.then(session => session.close());
        if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
        if (warningTimerRef.current) clearTimeout(warningTimerRef.current);
        scriptProcessorRef.current?.disconnect();
    };

    // --- Score Editing Handlers ---
    const handleEditScores = () => {
        setEditedScoreData(scoreData); // Copy current scores to editable state
        setIsEditingScores(true);
    };

    const handleCancelEditScores = () => {
        setEditedScoreData(null);
        setIsEditingScores(false);
    };

    const handleSaveScores = () => {
        if (editedScoreData) {
            setScoreData(editedScoreData); // Commit changes
        }
        setIsEditingScores(false);
        setEditedScoreData(null);
    };

    const handleScoreChange = (category: keyof Omit<ScoreData, 'overallScore'>, newScore: number) => {
        setEditedScoreData(prevData => {
            if (!prevData) return null;
            
            const newData: ScoreData = {
                ...prevData,
                [category]: {
                    ...prevData[category],
                    score: newScore,
                },
            };
            
            // Recalculate overall score
            const scores = [newData.rapport.score, newData.historyTaking.score, newData.communication.score];
            newData.overallScore = scores.reduce((a, b) => a + b, 0) / scores.length;

            return newData;
        });
    };

    if (examState === 'ended') {
        return (
          <div className="flex flex-col items-center justify-center h-[calc(100vh-68px)]">
            <div className="text-center p-10 bg-gray-900/50 rounded-2xl shadow-2xl border border-gray-700/50">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto mb-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-teal-400">
                {isBlocked ? 'Exam Terminated' : 'Exam Complete'}
              </h2>
              <p className="text-gray-300 mt-2">You may now close this window.</p>
            </div>
          </div>
        );
      }

    return (
        <div className="flex flex-col md:flex-row h-[calc(100vh-68px)]">
             <canvas ref={canvasRef} className="hidden"></canvas>
            <div className="flex-grow p-4 flex flex-col items-center justify-center gap-4">
                <div className="w-full max-w-4xl aspect-video bg-black rounded-xl overflow-hidden shadow-2xl border-2 border-blue-500/30 relative group">
                    <video ref={localVideoRef} autoPlay muted className="w-full h-full object-cover transform -scale-x-100" />
                    <div className="absolute top-2 left-2 bg-black/50 text-white px-3 py-1 rounded-full text-sm font-semibold border border-gray-700">You (Student)</div>
                    {unclearAudioWarning && (
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center pointer-events-none">
                            <div className="text-white bg-red-600/90 px-4 py-2 rounded-lg font-semibold flex items-center gap-2 shadow-lg">
                                 <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                                </svg>
                                Could not hear you clearly. Please repeat.
                            </div>
                        </div>
                    )}
                    {silenceWarningVisible && (
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center pointer-events-none">
                            <div className="text-white bg-amber-600/90 px-4 py-2 rounded-lg font-semibold flex items-center gap-2 shadow-lg">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                Session will end in 5 seconds due to inactivity.
                            </div>
                        </div>
                    )}
                     {examState === 'ready' && !isBlocked && (
                        <div className="absolute inset-0 bg-black/70 flex items-center justify-center backdrop-blur-sm">
                            <button onClick={handleStartExam} className="px-8 py-4 text-2xl font-bold text-gray-900 bg-green-500 rounded-lg hover:bg-green-600 transition-all transform hover:scale-110 shadow-2xl shadow-green-500/30 border-2 border-green-300">
                                Ready to Begin
                            </button>
                        </div>
                    )}
                    {examState === 'q&a' && !isBlocked && (
                        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20">
                            <button 
                                onClick={handleEndExam} 
                                className="px-6 py-3 font-bold text-white bg-red-600 rounded-lg hover:bg-red-700 transition-all transform hover:scale-105 shadow-2xl shadow-red-500/40 border-2 border-red-400 flex items-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                                End Exam
                            </button>
                        </div>
                    )}
                </div>
                <div className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-4">
                     <ParticipantCard name="Alex (Standardized Patient)" isSpeaking={isPatientSpeaking} isActive={examState === 'in-progress'}>
                        <div className="w-full h-full flex items-center justify-center p-4 text-center">
                            {isPatientSpeaking ? <AudioWaveform color="teal" /> : <StaticIcon type="patient" isActive={examState === 'in-progress'} />}
                        </div>
                     </ParticipantCard>
                     <ParticipantCard name="AI Examiner" isSpeaking={isExaminerSpeaking} isActive={examState === 'feedback' || examState === 'q&a'}>
                        <div className="w-full h-full flex flex-col items-center justify-center p-4 text-center gap-2">
                             {isExaminerSpeaking ? <AudioWaveform color="purple" /> : <StaticIcon type="examiner" isActive={examState === 'feedback' || examState === 'q&a'} />}
                            <p className="text-purple-300 font-semibold text-sm">Feedback:</p>
                            <p className="text-gray-200 italic text-sm leading-tight px-2 overflow-y-auto max-h-16">"{examinerFeedback}"</p>
                        </div>
                     </ParticipantCard>
                </div>
                {scoreData && (
                    <div className="w-full max-w-4xl mt-4">
                        <ScoreDisplay
                            scoreData={scoreData}
                            editedScoreData={editedScoreData}
                            isEditing={isEditingScores}
                            isVerified={isScoreVerified}
                            onEdit={handleEditScores}
                            onCancel={handleCancelEditScores}
                            onSave={handleSaveScores}
                            onScoreChange={handleScoreChange}
                            onVerify={() => setIsScoreVerified(true)}
                        />
                    </div>
                )}
            </div>
            {/* Transcript Sidebar */}
            <aside className="w-full md:w-96 bg-gray-900/50 border-l border-gray-700/50 flex flex-col">
                <h2 className="text-xl font-bold p-4 border-b border-gray-700/50 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">Live Transcript</h2>
                <div className="flex-grow p-4 overflow-y-auto space-y-4">
                    {transcripts.map((t, i) => (
                        <TranscriptBubble key={i} role={t.role} text={t.text} />
                    ))}
                    {currentStudentTranscript && <TranscriptBubble role="Student" text={currentStudentTranscript} isPartial />}
                    {currentSpTranscript && <TranscriptBubble role="SP" text={currentSpTranscript} isPartial />}
                    {currentExaminerTranscript && <TranscriptBubble role="Examiner" text={currentExaminerTranscript} isPartial />}
                    <div ref={transcriptEndRef} />
                </div>
                <div className="p-3 bg-black/30 border-t border-gray-700/50 text-center text-sm text-gray-400">
                    Status: <span className="font-semibold text-gray-300">{status}</span>
                </div>
            </aside>
        </div>
    );
};

// --- UI Components ---

const Star: React.FC<{ filled: boolean; interactive?: boolean; onClick?: () => void }> = ({ filled, interactive, onClick }) => (
    <svg 
        className={`w-5 h-5 ${filled ? 'text-yellow-400' : 'text-gray-600'} ${interactive ? 'cursor-pointer hover:scale-125 transition-transform' : ''}`} 
        xmlns="http://www.w3.org/2000/svg" 
        viewBox="0 0 20 20" 
        fill="currentColor"
        onClick={onClick}
    >
        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
    </svg>
);

const ScoreCategoryDisplay: React.FC<{ 
    name: string; 
    categoryKey: keyof Omit<ScoreData, 'overallScore'>;
    data: { score: number; justification: string };
    isEditing?: boolean;
    onScoreChange?: (category: keyof Omit<ScoreData, 'overallScore'>, score: number) => void;
}> = ({ name, categoryKey, data, isEditing, onScoreChange }) => (
    <div>
        <h4 className="font-semibold text-gray-200">{name}</h4>
        <div className="flex items-center gap-2 my-1">
            {[...Array(5)].map((_, i) => (
                <Star 
                    key={i} 
                    filled={i < data.score} 
                    interactive={isEditing}
                    onClick={isEditing ? () => onScoreChange?.(categoryKey, i + 1) : undefined}
                />
            ))}
            <span className="font-bold text-lg text-white">{data.score}/5</span>
        </div>
        <p className="text-sm italic text-gray-400">"{data.justification}"</p>
    </div>
);

const ScoreDisplay: React.FC<{ 
    scoreData: ScoreData; 
    editedScoreData: ScoreData | null;
    isEditing: boolean;
    isVerified: boolean; 
    onVerify: () => void;
    onEdit: () => void;
    onCancel: () => void;
    onSave: () => void;
    onScoreChange: (category: keyof Omit<ScoreData, 'overallScore'>, score: number) => void;
}> = ({ scoreData, editedScoreData, isEditing, isVerified, onVerify, onEdit, onCancel, onSave, onScoreChange }) => {
    const displayData = isEditing && editedScoreData ? editedScoreData : scoreData;

    return (
        <div className="bg-gray-900/70 rounded-lg p-6 border border-gray-700 backdrop-blur-sm animate-fade-in">
            <div className="flex flex-wrap justify-between items-start gap-4">
                <div>
                    <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500">
                        Performance Scorecard
                    </h3>
                    <p className="text-sm text-gray-400">AI-generated assessment based on performance.</p>
                </div>
                {isVerified ? (
                     <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-900/50 border border-green-500/50 text-green-300 font-semibold text-sm">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" /></svg>
                         Verified
                     </div>
                ) : (
                    <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-amber-900/50 border border-amber-500/50 text-amber-300 font-semibold text-sm">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM10 13a1 1 0 110-2 1 1 0 010 2zm-1-8a1 1 0 011-1h.008a1 1 0 011 1v3.008a1 1 0 01-1 1H9a1 1 0 01-1-1V5z" clipRule="evenodd" /></svg>
                        Pending Human Verification
                     </div>
                )}
            </div>
            <div className="mt-4 grid md:grid-cols-3 gap-6 border-t border-gray-700 pt-4">
                <ScoreCategoryDisplay name="Opening & Rapport" categoryKey="rapport" data={displayData.rapport} isEditing={isEditing} onScoreChange={onScoreChange} />
                <ScoreCategoryDisplay name="History Taking" categoryKey="historyTaking" data={displayData.historyTaking} isEditing={isEditing} onScoreChange={onScoreChange} />
                <ScoreCategoryDisplay name="Communication Skills" categoryKey="communication" data={displayData.communication} isEditing={isEditing} onScoreChange={onScoreChange} />
            </div>
            <div className="mt-6 border-t border-gray-700 pt-4 flex flex-wrap gap-4 justify-between items-center">
                 <div>
                    <p className="text-gray-400 text-sm">Overall Score</p>
                    <p className="text-3xl font-bold text-white">{displayData.overallScore.toFixed(1)} / 5.0</p>
                </div>
                <div className="flex items-center gap-3">
                    {!isVerified && !isEditing && (
                        <>
                            <button onClick={onEdit} className="px-4 py-2 font-semibold text-white bg-blue-600 rounded-md transition-all hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 focus:ring-offset-gray-900">
                                Edit Scores
                            </button>
                            <button onClick={onVerify} className="px-4 py-2 font-semibold text-gray-900 bg-green-500 rounded-md transition-all hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 focus:ring-offset-gray-900">
                                Verify Score
                            </button>
                        </>
                    )}
                     {isEditing && (
                        <>
                            <button onClick={onCancel} className="px-4 py-2 font-semibold text-white bg-gray-600 rounded-md transition-all hover:bg-gray-700">
                                Cancel
                            </button>
                            <button onClick={onSave} className="px-4 py-2 font-semibold text-gray-900 bg-green-500 rounded-md transition-all hover:bg-green-600">
                                Save Changes
                            </button>
                        </>
                    )}
                </div>
            </div>
            <style>{`
                @keyframes fade-in {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fade-in {
                    animation: fade-in 0.5s ease-out forwards;
                }
            `}</style>
        </div>
    );
};


const StaticIcon: React.FC<{ type: 'patient' | 'examiner', isActive: boolean }> = ({ type, isActive }) => {
    const color = type === 'patient' ? 'text-teal-400' : 'text-purple-400';
    const inactiveColor = 'text-gray-600';
    if (type === 'patient') return (
        <svg xmlns="http://www.w3.org/2000/svg" className={`h-16 w-16 transition-colors ${isActive ? color : inactiveColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
    );
    return (
        <svg xmlns="http://www.w3.org/2000/svg" className={`h-12 w-12 transition-colors ${isActive ? color : inactiveColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
        </svg>
    );
};

const AudioWaveform: React.FC<{color: 'teal' | 'purple'}> = ({color}) => (
    <div className="flex items-center justify-center space-x-1.5">
        {[...Array(5)].map((_, i) => (
            <div key={i} className={`w-1.5 bg-${color}-400 rounded-full animate-wave`} style={{ animationDelay: `${i * 100}ms`, height: `${12 + Math.random() * 24}px` }}></div>
        ))}
        <style>{`
            @keyframes wave {
                0%, 100% { transform: scaleY(0.5); }
                50% { transform: scaleY(1.5); }
            }
            .animate-wave {
                animation: wave 1.2s infinite ease-in-out;
            }
        `}</style>
    </div>
);

const ParticipantCard: React.FC<{ name: string; children: React.ReactNode; isSpeaking?: boolean; isActive: boolean }> = ({ name, children, isSpeaking, isActive }) => {
    const activeColor = name.includes('Patient') ? 'border-teal-400 shadow-lg shadow-teal-500/20' : 'border-purple-400 shadow-lg shadow-purple-500/20';
    return (
        <div className={`aspect-video bg-gray-900/70 rounded-lg flex flex-col items-center justify-center text-gray-400 border border-gray-700 overflow-hidden relative transition-all duration-300 backdrop-blur-sm
        ${isSpeaking ? activeColor : ''}
        ${!isActive ? 'opacity-50' : ''}
        `}>
            <div className="w-full h-full flex-grow">{children}</div>
            <p className="absolute bottom-0 w-full p-2 text-center bg-black/50 font-semibold text-sm">{name}</p>
        </div>
    )
};

const TranscriptBubble: React.FC<{ role: 'Student' | 'SP' | 'Examiner', text: string, isPartial?: boolean }> = ({ role, text, isPartial }) => {
    const isStudent = role === 'Student';
    
    const bubbleStyles = {
        Student: 'bg-gradient-to-br from-blue-600/30 to-blue-700/30',
        SP: 'bg-gradient-to-br from-teal-600/30 to-teal-700/30',
        Examiner: 'bg-gradient-to-br from-purple-600/30 to-purple-700/30',
    };

    const nameStyles = {
        Student: 'text-blue-300',
        SP: 'text-teal-300',
        Examiner: 'text-purple-300',
    };

    return (
         <div className={`w-full flex ${isStudent ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[90%] p-3 rounded-xl ${bubbleStyles[role]} ${isPartial ? 'opacity-70' : ''}`}>
                <p className={`font-bold text-sm mb-1 ${nameStyles[role]}`}>{role}</p>
                <p className="text-white text-base">{text}</p>
            </div>
        </div>
    )
};

export default Call;