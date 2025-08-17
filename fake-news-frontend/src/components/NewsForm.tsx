"use client";

import { useState, useEffect, useRef } from "react";
import "./NewsForm.css";

export default function NewsForm() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Refs for scroll animations
  const heroRef = useRef<HTMLDivElement>(null);
  const toolRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLDivElement>(null);
  const featuresRef = useRef<HTMLDivElement>(null);
  const howItWorksRef = useRef<HTMLDivElement>(null);
  const disclaimerRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const windowHeight = window.innerHeight;

      // Textarea sticky effect
      if (textareaRef.current) {
        const textareaRect = textareaRef.current.getBoundingClientRect();
        const textareaTop = textareaRect.top + scrollY;
        const stickyStart = textareaTop - windowHeight * 0.5;
        const stickyEnd = textareaTop + textareaRect.height;
        
        if (scrollY > stickyStart && scrollY < stickyEnd) {
          const progress = Math.min((scrollY - stickyStart) / (windowHeight * 0.3), 1);
          textareaRef.current.style.transform = `translateY(${(1 - progress) * 100}px) scale(${0.95 + progress * 0.05})`;
          textareaRef.current.style.opacity = `${0.7 + progress * 0.3}`;
        } else if (scrollY >= stickyEnd) {
          textareaRef.current.style.transform = 'translateY(0) scale(1)';
          textareaRef.current.style.opacity = '1';
        }
      }

      // Horizontal scroll effect for features
      if (featuresRef.current) {
        const featuresRect = featuresRef.current.getBoundingClientRect();
        if (featuresRect.top < windowHeight && featuresRect.bottom > 0) {
          const progress = Math.max(0, Math.min(1, (windowHeight - featuresRect.top) / windowHeight));
          const cards = featuresRef.current.querySelectorAll('.feature-card');
          cards.forEach((card, index) => {
            const element = card as HTMLElement;
            const delay = index * 0.2;
            const cardProgress = Math.max(0, Math.min(1, progress - delay));
            
            if (cardProgress > 0) {
              const translateX = (1 - cardProgress) * (index === 1 ? 100 : -100);
              element.style.transform = `translateX(${translateX}px) scale(${0.8 + cardProgress * 0.2})`;
              element.style.opacity = `${cardProgress}`;
            } else {
              // Reset to initial state
              const initialX = index === 1 ? 100 : -100;
              element.style.transform = `translateX(${initialX}px) scale(0.8)`;
              element.style.opacity = '0';
            }
          });
        }
      }

      // Stats counter animation
      if (statsRef.current) {
        const statsRect = statsRef.current.getBoundingClientRect();
        if (statsRect.top < windowHeight && !statsRef.current.classList.contains('counted')) {
          statsRef.current.classList.add('counted');
          animateCounters();
        }
      }
    };

    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };

    const observerCallback = (entries: IntersectionObserverEntry[]) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animate-in');
        }
      });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);
    
    const sections = [heroRef, toolRef, howItWorksRef, disclaimerRef];
    sections.forEach(ref => {
      if (ref.current) {
        observer.observe(ref.current);
      }
    });

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial call

    return () => {
      observer.disconnect();
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const animateCounters = () => {
    const counters = [
      { element: document.querySelector('.counter-accuracy'), target: 95.2, suffix: '%' },
      { element: document.querySelector('.counter-articles'), target: 70, suffix: 'K+' },
      { element: document.querySelector('.counter-uptime'), target: 24, suffix: '/7' }
    ];

    counters.forEach(({ element, target, suffix }) => {
      if (!element) return;
      let current = 0;
      const increment = target / 60;
      const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
          current = target;
          clearInterval(timer);
        }
        element.textContent = `${current.toFixed(target < 10 ? 0 : 1)}${suffix}`;
      }, 50);
    });
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      setResult(data.prediction);
    } catch (err) {
      setError("Error predicting: " + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Background Grid */}
      <div className="news-background-grid"></div>
      
      {/* Main Content Container */}
      <div className="min-h-screen py-12 px-4">
        {/* Hero Section */}
        <div ref={heroRef} className="scroll-animate max-w-6xl mx-auto mb-12">
          <div className="text-center mb-16">
            <div className="inline-block p-4 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 mb-6 scroll-fade-up hero-icon">
              <span className="text-4xl">🔍</span>
            </div>
            <h1 className="news-header text-5xl md:text-7xl mb-6 leading-tight scroll-fade-up scroll-delay-1">
              AI-Powered News
              <br />
              <span className="bg-gradient-to-r from-red-500 to-orange-500 bg-clip-text text-transparent">
                Authenticity
              </span> Detector
            </h1>
            <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 max-w-4xl mx-auto leading-relaxed scroll-fade-up scroll-delay-2">
              Combat misinformation with cutting-edge artificial intelligence. 
              Analyze news articles, social media posts, and any text content 
              to identify potentially misleading or false information.
            </p>
            
            <div ref={statsRef} className="stats-container flex justify-center items-center mt-8 space-x-8 scroll-fade-up scroll-delay-3">
              <div className="text-center scroll-scale stat-item">
                <div className="text-2xl font-bold text-blue-600 counter-accuracy">0%</div>
                <div className="text-sm text-gray-500">Accuracy Rate</div>
              </div>
              <div className="w-px h-12 bg-gray-300"></div>
              <div className="text-center scroll-scale scroll-delay-1 stat-item">
                <div className="text-2xl font-bold text-green-600 counter-articles">0K+</div>
                <div className="text-sm text-gray-500">Articles Analyzed</div>
              </div>
              <div className="w-px h-12 bg-gray-300"></div>
              <div className="text-center scroll-scale scroll-delay-2 stat-item">
                <div className="text-2xl font-bold text-purple-600 counter-uptime">0/7</div>
                <div className="text-sm text-gray-500">Real-time Analysis</div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Analysis Tool */}
        <div ref={toolRef} className="scroll-animate max-w-5xl mx-auto mb-16">
          <div className="news-card rounded-3xl p-10 scroll-fade-up tool-card">
            {/* Tool Header */}
            <div className="text-center mb-10">
              <h2 className="text-3xl font-bold mb-4 text-gray-800 dark:text-gray-100 scroll-fade-up">
                <span className="news-icon">🔍</span>
                News Analysis Tool
              </h2>
              <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-purple-600 mx-auto rounded-full scroll-expand"></div>
            </div>

            {/* Main Form */}
            <div className="space-y-8">
              <div ref={textareaRef} className="relative textarea-container">
                <label 
                  htmlFor="news-text" 
                  className="block text-lg font-semibold text-gray-700 dark:text-gray-200 mb-4"
                >
                  <span className="news-icon">✍️</span>
                  Enter News Article or Text Content
                </label>
                <div className="relative">
                  <textarea
                    id="news-text"
                    className="news-textarea w-full p-8 rounded-2xl resize-none focus:outline-none text-gray-800 dark:text-gray-100 text-lg leading-relaxed"
                    rows={10}
                    placeholder="Paste your news article, headline, social media post, or any text content here to analyze its authenticity. The more content you provide, the more accurate the analysis will be..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                  />
                  <div className="absolute bottom-4 right-6 flex items-center space-x-4 text-sm text-gray-400">
                    <span>{text.length} characters</span>
                    <span className="w-px h-4 bg-gray-300"></span>
                    <span>{text.split(/\s+/).filter(word => word.length > 0).length} words</span>
                  </div>
                </div>
              </div>

              <button
                onClick={() => handleSubmit()}
                className="news-button w-full py-6 px-10 rounded-2xl text-white font-bold text-xl transition-all duration-300 flex items-center justify-center scroll-fade-up scroll-delay-2"
                disabled={loading || !text.trim()}
              >
                {loading && <span className="loading-spinner"></span>}
                <span className="news-icon text-2xl">🚀</span>
                {loading ? "Analyzing Content..." : "Analyze for Authenticity"}
              </button>
            </div>

            {/* Results Section */}
            {result && (
              <div className={`result-card mt-10 p-8 rounded-2xl scroll-fade-up ${
                result === "REAL" ? "result-real" : "result-fake"
              }`}>
                <div className="text-center">
                  <div className="text-6xl mb-4 scroll-bounce">
                    {result === "REAL" ? "✅" : "❌"}
                  </div>
                  <h3 className="text-2xl font-bold mb-4">
                    Analysis Complete
                  </h3>
                  <p className={`text-4xl font-extrabold mb-4 ${
                    result === "REAL" 
                      ? "text-green-700 dark:text-green-300" 
                      : "text-red-700 dark:text-red-300"
                  }`}>
                    {result === "REAL" ? "LIKELY AUTHENTIC" : "POTENTIALLY FAKE"}
                  </p>
                  <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
                    {result === "REAL" 
                      ? "Our AI analysis suggests this content appears to be legitimate and trustworthy based on linguistic patterns, factual consistency, and writing style." 
                      : "Our AI has detected potential signs of misinformation, bias, or misleading content. We recommend verifying this information through multiple trusted sources."
                    }
                  </p>
                </div>
              </div>
            )}

            {/* Error Section */}
            {error && (
              <div className="error-card mt-8 p-8 rounded-2xl">
                <div className="flex items-center justify-center space-x-4">
                  <span className="text-4xl">⚠️</span>
                  <div className="text-center">
                    <h3 className="text-xl font-bold text-red-800 dark:text-red-200 mb-2">
                      Analysis Failed
                    </h3>
                    <p className="text-red-700 dark:text-red-300 text-lg">{error}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Features Section */}
        <div ref={featuresRef} className="features-section max-w-6xl mx-auto mb-16">
          <div className="grid md:grid-cols-3 gap-8">
            <div className="news-card rounded-2xl p-8 text-center feature-card">
              <div className="text-4xl mb-4 scroll-float">🤖</div>
              <h3 className="text-xl font-bold mb-4">AI-Powered Analysis</h3>
              <p className="text-gray-600 dark:text-gray-300">
                Advanced machine learning algorithms trained on thousands of articles 
                to detect patterns of misinformation and bias.
              </p>
            </div>
            <div className="news-card rounded-2xl p-8 text-center feature-card">
              <div className="text-4xl mb-4 scroll-float scroll-delay-2">🎯</div>
              <h3 className="text-xl font-bold mb-4">High Accuracy</h3>
              <p className="text-gray-600 dark:text-gray-300">
                Our model achieves over 95% accuracy in detecting 
                fake news and misleading content across various domains.
              </p>
            </div>
            <div className="news-card rounded-2xl p-8 text-center feature-card">
              <div className="text-4xl mb-4 scroll-float scroll-delay-1">⚡</div>
              <h3 className="text-xl font-bold mb-4">Real-time Results</h3>
              <p className="text-gray-600 dark:text-gray-300">
                Get instant analysis results in seconds. No waiting, 
                no complex setup - just paste and analyze.
              </p>
            </div>
            
          </div>
        </div>

        {/* How it Works Section */}
        <div ref={howItWorksRef} className="scroll-animate max-w-5xl mx-auto mb-16">
          <div className="news-card rounded-2xl p-10 scroll-fade-up">
            <h2 className="text-3xl font-bold text-center mb-10">
              <span className="news-icon">⚙️</span>
              How Our AI Detection Works
            </h2>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div className="space-y-6">
                <div className="flex items-start space-x-4 scroll-slide-right">
                  <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">1</div>
                  <div>
                    <h4 className="font-bold text-lg mb-2">Text Analysis</h4>
                    <p className="text-gray-600 dark:text-gray-300">
                      Our AI examines writing patterns, grammar, and linguistic structures to identify suspicious content.
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-4 scroll-slide-right scroll-delay-1">
                  <div className="bg-purple-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">2</div>
                  <div>
                    <h4 className="font-bold text-lg mb-2">Fact Checking</h4>
                    <p className="text-gray-600 dark:text-gray-300">
                      Cross-references claims against known facts and identifies potential inconsistencies.
                    </p>
                  </div>
                </div>
                <div className="flex items-start space-x-4 scroll-slide-right scroll-delay-2">
                  <div className="bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold">3</div>
                  <div>
                    <h4 className="font-bold text-lg mb-2">Bias Detection</h4>
                    <p className="text-gray-600 dark:text-gray-300">
                      Identifies emotional manipulation, loaded language, and other indicators of biased reporting.
                    </p>
                  </div>
                </div>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl p-8 scroll-slide-left">
                <div className="text-center">
                  <div className="text-6xl mb-4 scroll-pulse">🧠</div>
                  <h4 className="text-xl font-bold mb-4">Neural Network Processing</h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    Our deep learning model processes multiple layers of information 
                    simultaneously to provide accurate authenticity scores.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div ref={disclaimerRef} className="scroll-animate max-w-4xl mx-auto">
          <div className="news-card rounded-2xl p-8 scroll-fade-up">
            <div className="text-center">
              <h3 className="text-xl font-bold mb-4">
                <span className="news-icon">📋</span>
                Important Disclaimer
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                This tool provides AI-generated predictions based on text analysis patterns. 
                While our model achieves high accuracy, it should be used as a supplementary tool 
                alongside critical thinking and verification from multiple trusted sources.
              </p>
              <div className="flex justify-center items-center space-x-6 text-sm text-gray-500">
                <span className="scroll-fade-up disclaimer-item">🔒 Privacy Protected</span>
                <span className="scroll-fade-up scroll-delay-1 disclaimer-item">🚀 Powered by Advanced AI</span>
                <span className="scroll-fade-up scroll-delay-2 disclaimer-item">📊 Continuously Improving</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}