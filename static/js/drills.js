/**
 * Shared JavaScript components for pronunciation drills
 * Provides reusable functionality for mic recording, timers, and feedback
 */

class DrillComponents {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.timerInterval = null;
        this.timerDuration = 0;
        this.timerElement = null;
        this.audioElement = null;
        this.videoElement = null;
        this.stream = null;
    }

    /**
     * Initialize microphone recording
     * @param {Object} options - Configuration options
     * @param {Function} onStart - Callback when recording starts
     * @param {Function} onStop - Callback when recording stops
     * @param {Function} onError - Callback for errors
     */
    async startMic(options = {}) {
        try {
            const constraints = {
                audio: true,
                video: options.video || false,
                ...options.constraints
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            if (options.video && options.videoElement) {
                this.videoElement = document.getElementById(options.videoElement);
                if (this.videoElement) {
                    this.videoElement.srcObject = this.stream;
                    this.videoElement.play();
                }
            }

            this.mediaRecorder = new MediaRecorder(this.stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                
                if (this.audioElement) {
                    this.audioElement.src = audioUrl;
                }
                
                if (options.onStop) {
                    options.onStop(audioBlob, audioUrl);
                }
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Add visual feedback
            this.showRecordingIndicator();
            
            if (options.onStart) {
                options.onStart();
            }

        } catch (error) {
            console.error('Error accessing microphone:', error);
            if (options.onError) {
                options.onError(error);
            }
        }
    }

    /**
     * Stop microphone recording
     */
    stopMic() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.hideRecordingIndicator();
            
            // Stop all tracks
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
        }
    }
    
    /**
     * Get the current audio stream for real-time analysis
     */
    getStream() {
        return this.stream;
    }

    /**
     * Start a countdown timer
     * @param {number} duration - Duration in seconds
     * @param {string} elementId - ID of the element to display timer
     * @param {Function} onComplete - Callback when timer completes
     * @param {Function} onTick - Callback for each second
     */
    startTimer(duration, elementId, onComplete = null, onTick = null) {
        this.timerDuration = duration;
        this.timerElement = document.getElementById(elementId);
        
        if (!this.timerElement) {
            console.error('Timer element not found:', elementId);
            return;
        }

        this.updateTimerDisplay();
        
        this.timerInterval = setInterval(() => {
            this.timerDuration--;
            this.updateTimerDisplay();
            
            if (onTick) {
                onTick(this.timerDuration);
            }
            
            if (this.timerDuration <= 0) {
                this.stopTimer();
                if (onComplete) {
                    onComplete();
                }
            }
        }, 1000);
    }

    /**
     * Stop the current timer
     */
    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    /**
     * Update timer display with animations
     */
    updateTimerDisplay() {
        if (!this.timerElement) return;

        const minutes = Math.floor(this.timerDuration / 60);
        const seconds = this.timerDuration % 60;
        const timeString = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        
        this.timerElement.textContent = timeString;
        
        // Add visual effects based on remaining time
        if (this.timerDuration <= 10) {
            this.timerElement.classList.add('text-red-500', 'animate-pulse');
        } else if (this.timerDuration <= 30) {
            this.timerElement.classList.add('text-yellow-500');
        } else {
            this.timerElement.classList.remove('text-red-500', 'text-yellow-500', 'animate-pulse');
        }
    }

    /**
     * Show feedback buttons for self-rating
     * @param {string} containerId - ID of container to add buttons to
     * @param {Function} onRating - Callback when user rates
     */
    showFeedback(containerId, onRating = null) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="flex justify-center space-x-4 mb-4">
                <button id="thumbs-down" class="feedback-btn bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-full transition-all duration-200 transform hover:scale-110">
                    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M18 9.5a1.5 1.5 0 11-3 0v-3a1.5 1.5 0 013 0v3zM14 9.5a1.5 1.5 0 11-3 0v-3a1.5 1.5 0 013 0v3zM10 9.5a1.5 1.5 0 11-3 0v-3a1.5 1.5 0 013 0v3zM6 9.5a1.5 1.5 0 11-3 0v-3a1.5 1.5 0 013 0v3zM2 9.5a1.5 1.5 0 11-3 0v-3a1.5 1.5 0 013 0v3z"/>
                    </svg>
                    <span class="ml-2">Needs Work</span>
                </button>
                <button id="thumbs-up" class="feedback-btn bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-full transition-all duration-200 transform hover:scale-110">
                    <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M2 10.5a1.5 1.5 0 113 0v3a1.5 1.5 0 01-3 0v-3zM6 10.5a1.5 1.5 0 113 0v3a1.5 1.5 0 01-3 0v-3zM10 10.5a1.5 1.5 0 113 0v3a1.5 1.5 0 01-3 0v-3zM14 10.5a1.5 1.5 0 113 0v3a1.5 1.5 0 01-3 0v-3zM18 10.5a1.5 1.5 0 113 0v3a1.5 1.5 0 01-3 0v-3z"/>
                    </svg>
                    <span class="ml-2">Great!</span>
                </button>
            </div>
        `;

        // Add event listeners
        document.getElementById('thumbs-down').addEventListener('click', () => {
            this.handleFeedback('poor', onRating);
        });

        document.getElementById('thumbs-up').addEventListener('click', () => {
            this.handleFeedback('good', onRating);
        });
    }

    /**
     * Handle feedback selection
     * @param {string} rating - 'good' or 'poor'
     * @param {Function} callback - Callback function
     */
    handleFeedback(rating, callback) {
        // Add visual feedback
        const buttons = document.querySelectorAll('.feedback-btn');
        buttons.forEach(btn => {
            btn.classList.remove('ring-4', 'ring-blue-300');
            btn.classList.add('opacity-50');
        });

        const selectedBtn = document.getElementById(rating === 'good' ? 'thumbs-up' : 'thumbs-down');
        selectedBtn.classList.add('ring-4', 'ring-blue-300');
        selectedBtn.classList.remove('opacity-50');

        // Add zap effect
        this.showZapEffect(selectedBtn);

        if (callback) {
            callback(rating);
        }
    }

    /**
     * Show zap effect animation
     * @param {HTMLElement} element - Element to animate
     */
    showZapEffect(element) {
        element.classList.add('animate-bounce');
        setTimeout(() => {
            element.classList.remove('animate-bounce');
        }, 1000);
    }

    /**
     * Show recording indicator
     */
    showRecordingIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'recording-indicator';
        indicator.className = 'fixed top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-full animate-pulse z-50';
        indicator.innerHTML = `
            <div class="flex items-center">
                <div class="w-3 h-3 bg-white rounded-full mr-2 animate-pulse"></div>
                Recording...
            </div>
        `;
        document.body.appendChild(indicator);
    }

    /**
     * Hide recording indicator
     */
    hideRecordingIndicator() {
        const indicator = document.getElementById('recording-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    /**
     * Create spinning wheel animation
     * @param {string} containerId - Container ID
     * @param {Array} items - Items to display on wheel
     * @param {Function} onLand - Callback when wheel lands
     */
    createSpinningWheel(containerId, items, onLand = null) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="relative w-64 h-64 mx-auto">
                <div id="wheel" class="w-full h-full border-4 border-gray-300 rounded-full relative overflow-hidden">
                    ${items.map((item, index) => `
                        <div class="wheel-segment absolute w-full h-full" 
                             style="transform: rotate(${index * (360 / items.length)}deg); 
                                    background: linear-gradient(${index * (360 / items.length)}deg, 
                                    hsl(${index * (360 / items.length)}, 70%, 60%), 
                                    hsl(${(index + 1) * (360 / items.length)}, 70%, 60%));">
                            <div class="absolute inset-0 flex items-center justify-center text-white font-bold text-lg"
                                 style="transform: rotate(${-index * (360 / items.length)}deg);">
                                ${item}
                            </div>
                        </div>
                    `).join('')}
                </div>
                <div class="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-2">
                    <div class="w-0 h-0 border-l-4 border-r-4 border-b-8 border-transparent border-b-red-500"></div>
                </div>
            </div>
            <button id="spin-btn" class="mt-4 bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-full transition-all duration-200">
                Spin the Wheel!
            </button>
        `;

        const wheel = document.getElementById('wheel');
        const spinBtn = document.getElementById('spin-btn');
        let isSpinning = false;

        spinBtn.addEventListener('click', () => {
            if (isSpinning) return;
            
            isSpinning = true;
            spinBtn.disabled = true;
            spinBtn.textContent = 'Spinning...';

            const spins = 5 + Math.random() * 5; // 5-10 spins
            const degrees = spins * 360 + Math.random() * 360;
            
            wheel.style.transition = 'transform 3s cubic-bezier(0.25, 0.1, 0.25, 1)';
            wheel.style.transform = `rotate(${degrees}deg)`;

            setTimeout(() => {
                const segmentIndex = Math.floor((360 - (degrees % 360)) / (360 / items.length));
                const selectedItem = items[segmentIndex];
                
                isSpinning = false;
                spinBtn.disabled = false;
                spinBtn.textContent = 'Spin Again!';
                
                if (onLand) {
                    onLand(selectedItem, segmentIndex);
                }
            }, 3000);
        });
    }

    /**
     * Create drag and drop interface
     * @param {string} containerId - Container ID
     * @param {Array} symbols - Phonetic symbols
     * @param {Function} onDrop - Callback when item is dropped
     */
    createDragDropInterface(containerId, symbols, onDrop = null) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="grid grid-cols-2 gap-4">
                <div class="symbol-pool bg-gray-100 p-4 rounded-lg">
                    <h3 class="text-lg font-semibold mb-4">Phonetic Symbols</h3>
                    <div class="flex flex-wrap gap-2">
                        ${symbols.map(symbol => `
                            <div class="symbol-item bg-white p-3 rounded-lg shadow-md cursor-move hover:shadow-lg transition-shadow duration-200"
                                 draggable="true" data-symbol="${symbol}">
                                ${symbol}
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div class="drop-zone bg-blue-50 p-4 rounded-lg border-2 border-dashed border-blue-300">
                    <h3 class="text-lg font-semibold mb-4">Drop Zone</h3>
                    <div id="drop-area" class="min-h-32 flex flex-wrap gap-2">
                        <!-- Dropped items will appear here -->
                    </div>
                </div>
            </div>
        `;

        // Add drag and drop functionality
        const symbolItems = container.querySelectorAll('.symbol-item');
        const dropArea = document.getElementById('drop-area');

        symbolItems.forEach(item => {
            item.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', e.target.dataset.symbol);
                e.target.style.opacity = '0.5';
            });

            item.addEventListener('dragend', (e) => {
                e.target.style.opacity = '1';
            });
        });

        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropArea.classList.add('bg-blue-100');
        });

        dropArea.addEventListener('dragleave', () => {
            dropArea.classList.remove('bg-blue-100');
        });

        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.classList.remove('bg-blue-100');
            
            const symbol = e.dataTransfer.getData('text/plain');
            const droppedItem = document.createElement('div');
            droppedItem.className = 'dropped-symbol bg-white p-2 rounded shadow-md';
            droppedItem.textContent = symbol;
            droppedItem.dataset.symbol = symbol;
            
            dropArea.appendChild(droppedItem);
            
            if (onDrop) {
                onDrop(symbol, dropArea);
            }
        });
    }

    /**
     * Play audio with waveform visualization
     * @param {string} audioUrl - URL of audio to play
     * @param {string} canvasId - ID of canvas for waveform
     */
    async playAudioWithWaveform(audioUrl, canvasId) {
        const audio = new Audio(audioUrl);
        const canvas = document.getElementById(canvasId);
        
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaElementSource(audio);
        
        source.connect(analyser);
        analyser.connect(audioContext.destination);
        
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);
            
            ctx.fillStyle = 'rgb(200, 200, 200)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                barHeight = (dataArray[i] / 255) * canvas.height;
                
                ctx.fillStyle = `rgb(${barHeight + 100}, 50, 50)`;
                ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                
                x += barWidth + 1;
            }
        };
        
        audio.addEventListener('play', draw);
        audio.play();
        
        return audio;
    }

    /**
     * Clean up resources
     */
    cleanup() {
        this.stopMic();
        this.stopTimer();
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        this.hideRecordingIndicator();
    }
}

// Global instance
window.drillComponents = new DrillComponents();

// Utility functions
window.DrillUtils = {
    /**
     * Format time in MM:SS format
     */
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    },

    /**
     * Generate random MLK excerpt
     */
    getRandomMLKExcerpt() {
        const excerpts = [
            "I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
            "We hold these truths to be self-evident, that all men are created equal.",
            "Let freedom ring from the prodigious hilltops of New Hampshire.",
            "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin.",
            "Now is the time to make real the promises of democracy.",
            "We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead."
        ];
        return excerpts[Math.floor(Math.random() * excerpts.length)];
    },

    /**
     * Get phonetic tips for common mispronunciations
     */
    getPhoneticTips(word) {
        const tips = {
            'rise': "Emphasize the 'r' sound and make the 'i' long",
            'dream': "Pronounce the 'ea' as a long 'e' sound",
            'nation': "Stress the first syllable 'NA-tion'",
            'freedom': "Make the 'ee' sound clear and long",
            'children': "Pronounce 'ch' clearly, not 'sh'",
            'judged': "The 'dg' should sound like 'j' not 'g'"
        };
        return tips[word.toLowerCase()] || "Focus on clear articulation";
    }
};

// Initialize the drill components
const drillComponents = new DrillComponents();

// Additional utility functions for new drills

// 8. Metronome with Web Audio API
export function createMetronome(audioContext, tempo = 120) {
    let isPlaying = false;
    let intervalId = null;
    let currentTempo = tempo;
    
    const playTick = () => {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.1);
    };
    
    return {
        start: () => {
            if (isPlaying) return;
            isPlaying = true;
            const interval = 60000 / currentTempo; // Convert BPM to milliseconds
            intervalId = setInterval(playTick, interval);
        },
        stop: () => {
            if (!isPlaying) return;
            isPlaying = false;
            clearInterval(intervalId);
        },
        setTempo: (newTempo) => {
            currentTempo = newTempo;
            if (isPlaying) {
                this.stop();
                this.start();
            }
        },
        isPlaying: () => isPlaying
    };
}

// 9. Filler Detection (simulated)
export function simulateFillerDetection(text, fillerWords = ['um', 'uh', 'like', 'you know', 'so', 'well']) {
    const detectedFillers = [];
    const words = text.toLowerCase().split(/\s+/);
    
    words.forEach((word, index) => {
        if (fillerWords.includes(word)) {
            detectedFillers.push({
                word: word,
                position: index,
                timestamp: Date.now()
            });
        }
    });
    
    return detectedFillers;
}

// 10. Speed Control for Audio Playback
export function createSpeedController(audioElement, speeds = [0.5, 0.75, 1.0, 1.25, 1.5]) {
    let currentSpeedIndex = 2; // Default to 1.0x
    
    const setSpeed = (speedIndex) => {
        if (speedIndex >= 0 && speedIndex < speeds.length) {
            currentSpeedIndex = speedIndex;
            audioElement.playbackRate = speeds[speedIndex];
        }
    };
    
    const nextSpeed = () => {
        setSpeed((currentSpeedIndex + 1) % speeds.length);
        return speeds[currentSpeedIndex];
    };
    
    const prevSpeed = () => {
        setSpeed((currentSpeedIndex - 1 + speeds.length) % speeds.length);
        return speeds[currentSpeedIndex];
    };
    
    return {
        setSpeed,
        nextSpeed,
        prevSpeed,
        getCurrentSpeed: () => speeds[currentSpeedIndex],
        getCurrentIndex: () => currentSpeedIndex
    };
}

// 11. Beat Pattern Generator
export function createBeatPattern(audioContext, pattern = [1, 0, 1, 0], bpm = 120) {
    let isPlaying = false;
    let intervalId = null;
    let patternIndex = 0;
    
    const playBeat = () => {
        if (pattern[patternIndex]) {
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.setValueAtTime(200, audioContext.currentTime);
            gainNode.gain.setValueAtTime(0.2, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.2);
        }
        
        patternIndex = (patternIndex + 1) % pattern.length;
    };
    
    return {
        start: () => {
            if (isPlaying) return;
            isPlaying = true;
            const interval = 60000 / (bpm * pattern.length / 4); // Adjust for pattern length
            intervalId = setInterval(playBeat, interval);
        },
        stop: () => {
            if (!isPlaying) return;
            isPlaying = false;
            clearInterval(intervalId);
            patternIndex = 0;
        },
        setBPM: (newBpm) => {
            bpm = newBpm;
            if (isPlaying) {
                this.stop();
                this.start();
            }
        },
        isPlaying: () => isPlaying
    };
}

// 12. Progress Bar Animation
export function animateProgressBar(elementId, targetPercentage, duration = 1000) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const startPercentage = parseInt(element.style.width) || 0;
    const startTime = Date.now();
    
    const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentPercentage = startPercentage + (targetPercentage - startPercentage) * progress;
        element.style.width = currentPercentage + '%';
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    };
    
    requestAnimationFrame(animate);
}

// 13. Zap Effect Animation
export function createZapEffect(element) {
    element.classList.add('zap-effect');
    setTimeout(() => {
        element.classList.remove('zap-effect');
    }, 500);
}

// 14. Streak Counter
export function createStreakCounter(elementId) {
    let streak = 0;
    const element = document.getElementById(elementId);
    
    return {
        increment: () => {
            streak++;
            if (element) {
                element.textContent = streak;
                element.classList.add('animate-pulse');
                setTimeout(() => element.classList.remove('animate-pulse'), 300);
            }
            return streak;
        },
        reset: () => {
            streak = 0;
            if (element) {
                element.textContent = streak;
            }
            return streak;
        },
        get: () => streak
    };
}

// 15. Word Counter
export function createWordCounter(elementId) {
    let wordCount = 0;
    const element = document.getElementById(elementId);
    
    return {
        increment: (words = 1) => {
            wordCount += words;
            if (element) {
                element.textContent = wordCount;
            }
            return wordCount;
        },
        reset: () => {
            wordCount = 0;
            if (element) {
                element.textContent = wordCount;
            }
            return wordCount;
        },
        get: () => wordCount
    };
}
