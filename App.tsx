import React, { useState, useEffect, useCallback } from 'react';
import WebcamProctor from './components/WebcamProctor';
import Lobby from './components/Lobby';

const App: React.FC = () => {
  const [examStarted, setExamStarted] = useState(false);
  const [fullscreenWarnings, setFullscreenWarnings] = useState(0);
  const [proctoringWarnings, setProctoringWarnings] = useState(0);
  const [isExamBlocked, setIsExamBlocked] = useState(false);
  const [warningMessage, setWarningMessage] = useState('');

  const totalWarnings = fullscreenWarnings + proctoringWarnings;

  const handleStartExam = () => {
    document.documentElement.requestFullscreen().catch(err => {
      console.error(`Error attempting to enable full-screen mode: ${err.message} (${err.name})`);
    });
    setExamStarted(true);
  };

  const showWarning = (message: string) => {
    setWarningMessage(message);
    setTimeout(() => setWarningMessage(''), 3000); // Hide after 3 seconds
  };
  
  const handleFullscreenChange = useCallback(() => {
    if (!document.fullscreenElement && examStarted && !isExamBlocked) {
      const newCount = fullscreenWarnings + 1;
      setFullscreenWarnings(newCount);
      showWarning(`Fullscreen exited. Warning ${totalWarnings + 1} of 5.`);
      if (newCount >= 2) {
          setIsExamBlocked(true);
          setWarningMessage('Exam locked. You have exited fullscreen mode multiple times.');
      }
    }
  }, [examStarted, isExamBlocked, fullscreenWarnings, totalWarnings]);

  const handleProctoringViolation = useCallback((reason: string) => {
      if (isExamBlocked) return;
      const newCount = proctoringWarnings + 1;
      setProctoringWarnings(newCount);
      showWarning(`Proctoring Alert: ${reason}. Warning ${totalWarnings + 1} of 5.`);
  }, [isExamBlocked, proctoringWarnings, totalWarnings]);

  useEffect(() => {
    if (totalWarnings >= 5 && !isExamBlocked) {
        setIsExamBlocked(true);
        setWarningMessage('Exam terminated after 5 warnings.');
    }
  }, [totalWarnings, isExamBlocked]);

  useEffect(() => {
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [handleFullscreenChange]);

  return (
    <div className="min-h-screen font-sans">
      <header className="bg-gray-900 p-4 sticky top-0 z-50">
        <h1 className="text-[35px] font-bold text-center text-white uppercase">
          Atomic Examiner OSCE Exam Simulator
        </h1>
      </header>
      <main className="relative">
        {!examStarted ? (
          <Lobby onStartExam={handleStartExam} />
        ) : (
          <WebcamProctor 
            onViolation={handleProctoringViolation}
            isBlocked={isExamBlocked}
          />
        )}
        
        {/* Warning & Block Overlay */}
        {(warningMessage || isExamBlocked) && (
            <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-40 backdrop-blur-sm">
                <div className={`p-8 rounded-2xl shadow-2xl border ${isExamBlocked ? 'border-red-500 bg-gray-900' : 'border-amber-500 bg-gray-800'} max-w-md text-center`}>
                     <svg xmlns="http://www.w3.org/2000/svg" className={`h-16 w-16 mx-auto mb-4 ${isExamBlocked ? 'text-red-500' : 'text-amber-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <h2 className="text-2xl font-bold mb-2">{isExamBlocked ? 'Exam Terminated' : 'Warning'}</h2>
                    <p className="text-lg text-gray-300">{warningMessage}</p>
                    {!isExamBlocked && <p className="text-sm text-gray-400 mt-4">Total Warnings: {totalWarnings} / 5</p>}
                </div>
            </div>
        )}
      </main>
    </div>
  );
};

export default App;