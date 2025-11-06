export interface ProctoringResult {
  isCheating: boolean;
  reason: string;
}

export interface TranscriptEntry {
    role: 'Student' | 'SP' | 'Examiner';
    text: string;
}

export interface ScoreCategory {
    score: number;
    justification: string;
}

export interface ScoreData {
    rapport: ScoreCategory;
    historyTaking: ScoreCategory;
    communication: ScoreCategory;
    overallScore: number;
}
