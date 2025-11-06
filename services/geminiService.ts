import { GoogleGenAI, Type, Modality } from "@google/genai";
import { ProctoringResult, TranscriptEntry, ScoreData } from '../types';

export const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

const proctoringPrompt = `
You are an advanced AI proctor for a secure online exam. Your task is to analyze the provided image of a student and determine if there is any suspicious behavior.

Check for the following potential cheating behaviors strictly:
1.  **Student Presence**: Is the student clearly visible and facing the screen? The student must be present in the frame.
2.  **Multiple People**: Is there more than one person in the camera's view?
3.  **Unauthorized Objects**: Can you see any prohibited items like smartphones, textbooks, or notes? **Note: Standard headphones or earbuds are PERMITTED and should NOT be flagged as unauthorized objects.**
4.  **Speaking**: Does it appear the student is talking or whispering to someone? (Note: The student is expected to speak as part of the exam, but you should flag if they seem to be speaking to someone off-camera).

Based on your analysis, you must respond with a JSON object. The JSON object must conform to the provided schema. It should have two keys:
- "isCheating": A boolean value. Set to 'true' if ANY suspicious behavior is detected, otherwise 'false'.
- "reason": A brief, clear string explaining the detected behavior if "isCheating" is true. If this "isCheating" is false, this string MUST be "All clear".

Examples of reasons:
- "Student not present in the frame."
- "Another person detected in the background."
- "Possible use of a mobile phone detected."
- "Student appears to be talking to someone off-camera."
`;

export const analyzeStudentFrame = async (base64ImageData: string): Promise<ProctoringResult> => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        parts: [
          { inlineData: { mimeType: 'image/jpeg', data: base64ImageData } },
          { text: proctoringPrompt }
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            isCheating: { type: Type.BOOLEAN, description: "Whether suspicious behavior is detected." },
            reason: { type: Type.STRING, description: "Reason for the detection or 'All clear'." },
          },
          required: ['isCheating', 'reason']
        }
      }
    });

    const jsonString = response.text.trim();
    const result: ProctoringResult = JSON.parse(jsonString);
    return result;
  } catch (error) {
    console.error("Error analyzing frame:", error);
    // Return a non-cheating result to avoid penalizing for API errors, but provide feedback.
    return { isCheating: false, reason: "AI analysis failed. Check console for details." };
  }
};


const detailedExaminerFeedbackPrompt = `You are a clinical exam proctor observing a simulated interaction between a medical student and a standardized patient. The interaction phase is now over. Your role is to provide comprehensive, constructive feedback to the student based on their performance.

Below is the full transcript of the conversation:
---
{transcript}
---

Based on this transcript, provide detailed feedback on the student's performance. Structure your feedback into the following sections:
1.  **Opening and Rapport:** How well did the student initiate the conversation and build a connection with the patient?
2.  **History Taking:** Evaluate the student's questioning technique. Did they use a mix of open and closed-ended questions? Was their approach logical? Did they explore all relevant aspects of the chief complaint?
3.  **Communication Skills:** Comment on the student's empathy, clarity, and overall professionalism.
4.  **Overall Summary and Suggestions:** Provide a summary of their strengths and offer specific, actionable suggestions for improvement.

Your response should be formatted for clarity and spoken aloud. Start with "Alright, let's review your performance."

After providing the full text-based feedback, you MUST provide a structured JSON object containing a quantitative assessment. This JSON object must be enclosed between "SCORE_JSON_START" and "SCORE_JSON_END" tags.

The JSON object should follow this structure:
{
  "rapport": { "score": <1-5>, "justification": "<brief reasoning>" },
  "historyTaking": { "score": <1-5>, "justification": "<brief reasoning>" },
  "communication": { "score": <1-5>, "justification": "<brief reasoning>" },
  "overallScore": <average score, 1 decimal place>
}

Scoring Rubric (1-5 scale):
- 5: Excellent - Demonstrates all criteria flawlessly.
- 4: Good - Demonstrates most criteria well with minor room for improvement.
- 3: Competent - Demonstrates foundational skills but has clear areas for improvement.
- 2: Needs Improvement - Misses key criteria or makes significant errors.
- 1: Unsatisfactory - Fails to demonstrate basic competency.
`;

export const streamDetailedExaminerFeedback = async (
    transcript: TranscriptEntry[],
    onChunk: (textChunk: string) => void,
): Promise<{ feedbackText: string; scoreData: ScoreData | null; }> => {
    if (transcript.length < 2) {
        const shortFeedback = "The interaction was too brief to provide detailed feedback.";
        onChunk(shortFeedback);
        return { feedbackText: shortFeedback, scoreData: null };
    }

    const formattedTranscript = transcript.map(t => `${t.role}: ${t.text}`).join('\n');
    const promptWithTranscript = detailedExaminerFeedbackPrompt.replace('{transcript}', formattedTranscript);
    
    let fullResponseText = '';
    let feedbackSoFar = '';
    const scoreStartTag = 'SCORE_JSON_START';

    try {
        const responseStream = await ai.models.generateContentStream({
            model: 'gemini-2.5-pro',
            contents: promptWithTranscript,
        });

        for await (const chunk of responseStream) {
            const text = chunk.text;
            if (text) {
                fullResponseText += text;
                
                if (fullResponseText.includes(scoreStartTag)) {
                    const feedbackPart = fullResponseText.split(scoreStartTag)[0];
                    const newChunkForUI = feedbackPart.substring(feedbackSoFar.length);
                    if (newChunkForUI) {
                        onChunk(newChunkForUI);
                    }
                    feedbackSoFar = feedbackPart;
                } else {
                    onChunk(text);
                    feedbackSoFar += text;
                }
            }
        }

        const scoreRegex = /SCORE_JSON_START([\s\S]*?)SCORE_JSON_END/;
        const match = fullResponseText.match(scoreRegex);
        
        let scoreData: ScoreData | null = null;
        let feedbackText = feedbackSoFar.trim();

        if (match && match[1]) {
            try {
                scoreData = JSON.parse(match[1].trim());
            } catch (e) {
                console.error("Failed to parse score JSON", e);
            }
        }
        
        return { feedbackText, scoreData };
    } catch (error) {
        console.error("Error streaming detailed examiner feedback:", error);
        const errorMsg = "Could not retrieve detailed examiner feedback due to an error.";
        onChunk(errorMsg);
        return { feedbackText: errorMsg, scoreData: null };
    }
};

export const getExaminerQAPrompt = (transcript: TranscriptEntry[], feedback: string): string => `You are an AI Examiner in a clinical simulation. You have just provided the following detailed feedback to a medical student based on their interaction with a standardized patient:

--- FEEDBACK DELIVERED ---
${feedback}
--- END FEEDBACK ---

The full transcript of the student-patient interaction is below for your reference:
--- TRANSCRIPT ---
${transcript.map(t => `${t.role}: ${t.text}`).join('\n')}
--- END TRANSCRIPT ---

The student may now ask you questions to clarify the feedback. Your goal is to be helpful, encouraging, and provide specific examples from the transcript if asked. Keep your answers concise and focused on the student's questions.

This is the start of the Q&A session. Your first response should be: "Do you have any questions about the feedback?"
`;


export const textToSpeech = async (text: string, voice: 'Zephyr' | 'Kore' = 'Zephyr'): Promise<string | null> => {
    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash-preview-tts",
            contents: [{ parts: [{ text }] }],
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: voice },
                    },
                },
            },
        });
        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        return base64Audio || null;
    } catch (error) {
        console.error("Error in text-to-speech:", error);
        return null;
    }
};