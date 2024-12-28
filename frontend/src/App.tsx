import { useState } from 'react';
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Loader2 } from "lucide-react";

interface AnalysisResults {
  captions: string[];
  concepts: string[];
  summary: string;
}

function App() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze() {
    if (!url) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          video_url: url,
          num_frames: 5,
          model: "gpt-4",
          temperature: 0.7
        })
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed. Please try again.');
      }
      
      const data = await response.json();
      setAnalysis(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-2">Video Analysis</h1>
        <p className="text-gray-400 text-center mb-8">
          Extract insights and analyze content from any video
        </p>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle>Analyze Video</CardTitle>
            <CardDescription className="text-gray-400">
              Enter a YouTube URL or video link to begin analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              <Input
                type="text"
                value={url}
                onChange={e => setUrl(e.target.value)}
                placeholder="https://youtube.com/watch?v=..."
                className="flex-1 bg-gray-900 border-gray-700"
              />
              <Button 
                onClick={handleAnalyze}
                disabled={loading || !url}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze'
                )}
              </Button>
            </div>

            {error && (
              <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
                {error}
              </div>
            )}
          </CardContent>
        </Card>

        {analysis && (
          <div className="mt-8 space-y-6">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle>Summary</CardTitle>
                <CardDescription className="text-gray-400">
                  AI-generated summary of the video content
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-300">{analysis.summary}</p>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle>Key Concepts</CardTitle>
                  <CardDescription className="text-gray-400">
                    Main themes and objects detected
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {analysis.concepts.map((concept, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-blue-900/50 border border-blue-700 rounded-full text-sm"
                      >
                        {concept}
                      </span>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gray-800 border-gray-700">
                <CardHeader>
                  <CardTitle>Frame Captions</CardTitle>
                  <CardDescription className="text-gray-400">
                    Descriptions of key frames
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {analysis.captions.map((caption, index) => (
                      <li key={index} className="text-gray-300">
                        <span className="font-semibold text-blue-400">Frame {index + 1}:</span>{' '}
                        {caption}
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
